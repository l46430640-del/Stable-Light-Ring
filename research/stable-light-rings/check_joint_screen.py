"""Bounded optical/radial screening along a solitonic coupling continuation.

Coupling changes only select backgrounds. Every tangent, radiation response
and radial energy variation is evaluated at fixed coupling. Positive trial
energy does not certify radial stability.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import scipy
import sympy as sp

from boson_star import boundary, potential, solve_background
from check_branch_response import light_ring_brackets
from concentrated_axial import ConcentratedResponse, bottom_action_variation
from radial_energy import RadialEnergy
from static_transversality import BranchTangent, background_charge, bottom_barrier_action
from trapped_axial import Background, gauss_mesh


DATA = Path(__file__).parent/"data/solitonic-screen"
SCALES = (.08, .0775, .075, .0725, .07, .0675, .065, .0625, .06)


class Predictor:
    """Thin-wall rescaling is an initial guess, not a symmetry of the theory."""
    def __init__(self, background, scale):
        self.background = background
        self.length = (background.scale/scale)**2
        self.frequency = scale/background.scale

    def sol(self, r):
        bg = self.background
        x = np.asarray(r)/self.length
        y = bg.sol(np.minimum(x, bg.radius)).copy()
        outer = x > bg.radius
        if np.any(outer):
            kappa = np.sqrt(1-bg.omega**2)
            beta = bg.mass*(2*bg.omega**2-1)/kappa
            a = bg.sol(bg.radius)[0]*np.exp(-kappa*(x[outer]-bg.radius))*(x[outer]/bg.radius)**(beta-1)
            y[0, outer] = a
            y[1, outer] = a*(-kappa+(beta-1)/x[outer])
            y[2, outer] = bg.mass/x[outer]
            y[3, outer] = np.log(bg.omega)-.5*np.log1p(-2*bg.mass/x[outer])
        y[1] /= self.length
        y[3] += np.log(self.frequency)
        return y


def path_amplitude(scale):
    fraction = (scale*scale-.06**2)/(.08**2-.06**2)
    ratio2 = (.044/.06)**2+fraction*((.06/.08)**2-(.044/.06)**2)
    return float(scale*np.sqrt(ratio2))


def equilibrium(previous, scale, central, radius=None, tolerance=2e-8):
    radius = float(40*(.08/scale)**2 if radius is None else radius)
    solution = solve_background(central=central, scale=scale, radius=radius,
                                tolerance=tolerance, max_nodes=4000,
                                previous=Predictor(previous, scale))
    if not solution.success:
        raise RuntimeError(f"Background stopped at sigma={scale}, A0={central}, "
                           f"nodes={solution.x.size}/4000: {solution.message}")
    assert solution.x.size <= 4000
    assert np.min(solution.y[0]) > 0 and np.min(1-2*solution.y[2]) > .2
    assert np.max(np.abs(boundary(solution.y[:, 0], solution.y[:, -1], radius))) < 1e-10
    end = solution.sol(radius)
    model = SimpleNamespace(sol=solution.sol, scale=float(scale), central=float(central),
                            radius=radius, mass=float(radius*end[2]),
                            omega=float(np.exp(end[3])*np.sqrt(1-2*end[2])))
    assert 0 < model.omega < 1
    r, a, b, u, zeta = solution.x, *solution.y
    V, _ = potential(central*a, scale)
    pressure = (np.exp(2*zeta)*(central*a)**2+(1-2*u)*(central*b)**2-V)/2
    H = 3*u+4*np.pi*r*r*pressure-1
    crossings = int(np.count_nonzero(H[:-1]*H[1:] < 0))
    bg = None
    if crossings == 2:
        bg = Background(solution=solution, central=central, scale=scale,
                        ring_brackets=light_ring_brackets(solution, central, scale))
    return solution, model, bg, crossings


def coupling_work(background):
    r, weights = gauss_mesh(background.sol.x)
    g = background.fields(r)
    s, sigma = g["A"]**2, background.scale
    derivative = 8*s*s/sigma**3*(1-2*s/sigma**2)
    return float(weights@(2*np.pi*r*r*g["alpha"]/np.sqrt(g["f"])*derivative))


def screening(background, tolerance=3e-6, direct=False):
    bg = background
    tangent = BranchTangent(bg, tolerance=tolerance)
    free = ConcentratedResponse(bg, tolerance=tolerance, fixed_charge=False)
    Np = float(free.state(bg.radius)[4])
    Tp, outer = free.limiting_action()
    dS = bottom_action_variation(bg, tangent.metric)[0]
    dN = float(tangent.charge_response)
    if abs(dN) < 1e-6:
        raise ValueError("The sampled tangent is too close to a charge extremum for this screen.")
    compensation = -Np*dS/dN
    energy = RadialEnergy(bg, tangent).variations()
    record = {"sigma": bg.scale, "central_amplitude": bg.central,
              "mass": bg.mass, "charge": background_charge(bg), "omega": bg.omega,
              "outer_radius": bg.radius, "light_rings": [bg.rs, bg.ru],
              "relative_well_depth": float((bg.Uu-bg.Us)/bg.Us),
              "unperturbed_action": bottom_barrier_action(bg), "outer_turning_point": outer,
              "auxiliary_action": Tp, "charge_adjustment_action": float(compensation),
              "fixed_charge_action": float(Tp+compensation),
              "dN_dA0_at_fixed_sigma": dN/bg.central,
              "dS_dA0_at_fixed_sigma": dS/bg.central,
              "coupling_work": coupling_work(bg),
              "radial_energy": energy,
              "tangent_first_law_error": tangent.diagnostics()["first_law_error"],
              "source_first_law_error": float(abs(bg.radius*free.state(bg.radius)[2]-1-bg.omega*Np)),
              "action_quadrature_change": abs(free.limiting_action(240)[0]-Tp),
              "response_tolerance": tolerance, "tangent_nodes": int(tangent.solution.x.size),
              "response_radial_nodes": int(2*free.solution.x.size)}
    assert record["action_quadrature_change"] < 2e-5
    assert record["tangent_first_law_error"] < 2e-5 and record["source_first_law_error"] < 2e-5
    assert energy["projection_discrepancy"] < 2e-4*(1+abs(energy["projected_hessian"]))
    assert energy["background_mass_reconstruction_error"] < 2e-7
    assert abs(energy["integrated_charge_variation"]) < 2e-5
    assert energy["charge_scaling_first_law_error"] < 2e-7
    assert energy["charge_scaling_upper_gap"] > 0
    if direct:
        fixed = ConcentratedResponse(bg, tolerance=tolerance)
        action = fixed.limiting_action()[0]
        record["direct_fixed_charge"] = {"action": action,
            "projection_difference": abs(action-record["fixed_charge_action"]),
            "added_mass": float(bg.radius*fixed.state(bg.radius)[2]),
            "charge_response": float(fixed.state(bg.radius)[4]),
            "radial_nodes": int(2*fixed.solution.x.size)}
        assert record["direct_fixed_charge"]["projection_difference"] < 2e-5
        assert abs(record["direct_fixed_charge"]["added_mass"]-1) < 2e-5
        assert abs(record["direct_fixed_charge"]["charge_response"]) < 1e-9
    return record


def check_formulas():
    s, sigma = sp.symbols("s sigma", positive=True)
    V = s*(1-2*s/sigma**2)**2
    assert sp.simplify(sp.diff(V, sigma)-8*s*s/sigma**3*(1-2*s/sigma**2)) == 0
    bg = Background()
    r = np.array([0., 1., 15., 30., 40.])
    assert np.max(np.abs(Predictor(bg, bg.scale).sol(r)-bg.sol(r))) == 0
    scale = .075
    prediction = Predictor(bg, scale)
    y = prediction.sol(r*prediction.length)
    expected = bg.sol(r).copy()
    expected[1] /= prediction.length
    expected[3] += np.log(scale/bg.scale)
    assert np.allclose(y, expected, rtol=1e-14, atol=1e-14)
    extended = Predictor(bg, bg.scale).sol(np.array([40., 45., 50.]))
    assert np.all(np.isfinite(extended)) and np.all(extended[0] > 0)
    assert abs(path_amplitude(.08)-.06) < 1e-16 and abs(path_amplitude(.06)-.044) < 1e-16
    probe = ConcentratedResponse.__new__(ConcentratedResponse)
    probe.background = SimpleNamespace(radius=56.186556927297666)
    probe.location = 22.29173643023918
    probe.widths = (probe.location, probe.background.radius-probe.location)
    assert probe.location+probe.widths[1] > probe.background.radius
    seen = []
    def homogeneous(r, y, include_singular=True):
        seen.append(r.copy())
        return np.zeros_like(y)
    probe.homogeneous = homogeneous
    probe.equation(np.array([0., .5, 1.]), np.zeros((10, 3)))
    assert seen[1][-1] == probe.background.radius
    try:
        bg.fields(np.nextafter(bg.radius, np.inf))
    except ValueError:
        pass
    else:
        raise AssertionError("The background domain check must remain strict.")
    print("PASS: coupling derivative, predictor identity/scaling/tail and continuation endpoints", flush=True)


def endpoint_checks(background, baseline, report, output):
    checks = report["endpoint_checks"] = {}
    for name, radius in (("tighter_tolerances", background.radius), ("outer_radius_60", 60.)):
        report["current_check"] = name
        save(report, output)
        solution, _, bg, count = equilibrium(background, .06, .044, radius, tolerance=1e-8)
        assert count == 2
        item = screening(bg, tolerance=1e-6, direct=True)
        item["background_nodes"] = int(solution.x.size)
        item["background_tolerance"] = 1e-8
        item["action_change_from_baseline"] = abs(item["fixed_charge_action"]-baseline["fixed_charge_action"])
        item["radial_energy_change_from_baseline"] = abs(item["radial_energy"]["direct_mass_second_variation"]
                                                        -baseline["radial_energy"]["direct_mass_second_variation"])
        assert item["action_change_from_baseline"] < 2e-5
        assert item["radial_energy_change_from_baseline"] < 2e-4
        checks[name] = item
        save(report, output)
        print(f"PASS: endpoint {name}, action={item['fixed_charge_action']:.10g}", flush=True)

    work = coupling_work(background)
    differences = checks["coupling_first_law"] = {"integral": work, "samples": []}
    for h in (1e-5, 5e-6):
        report["current_check"] = f"coupling difference h={h}"
        save(report, output)
        values, nodes = [], []
        for sign in (-1, 1):
            solution, _, bg, count = equilibrium(background, .06+sign*h, .044,
                                                  background.radius, tolerance=1e-8)
            assert count == 2
            values.append((bg.mass, background_charge(bg)))
            nodes.append(int(solution.x.size))
        dM, dN = (np.array(values[1])-np.array(values[0]))/(2*h)
        result = float(dM-background.omega*dN)
        error = abs(result-work)
        differences["samples"].append({"step": h, "dM_dsigma_at_fixed_A0": float(dM),
            "dN_dsigma_at_fixed_A0": float(dN), "dM_minus_omega_dN": result,
            "absolute_error": error, "background_nodes": nodes, "background_tolerance": 1e-8})
        assert error < .01
        save(report, output)
        print(f"PASS: coupling first law h={h:g}, error={error:.6g}", flush=True)
    coarse, fine = differences["samples"]
    differences["error_reduction"] = coarse["absolute_error"]/fine["absolute_error"]
    extrapolated = (4*fine["dM_minus_omega_dN"]-coarse["dM_minus_omega_dN"])/3
    differences["richardson_error"] = abs(extrapolated-work)
    assert 3.8 < differences["error_reduction"] < 4.2
    assert differences["richardson_error"] < 1e-5
    save(report, output)


def save(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


def verification(output):
    if not __debug__:
        raise RuntimeError("Run without -O: numerical screening requires assertions.")
    start = perf_counter()
    check_formulas()
    previous = Background()
    seed = Path(__file__).parent/"data/s08a06/background.npz"
    report = {"status": "running", "scope": "Numerical joint screen, not an exact sign or radial stability certificate.",
              "seed_sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
              "software": {"numpy": np.__version__, "scipy": scipy.__version__, "sympy": sp.__version__},
              "max_radial_points": 4000, "samples": []}
    try:
        for scale in SCALES:
            central = path_amplitude(scale)
            report["current_parameters"] = {"sigma": scale, "central_amplitude": central}
            save(report, output)
            solution, previous, bg, count = equilibrium(previous, scale, central)
            item = {"sigma": scale, "central_amplitude": central, "background_nodes": int(solution.x.size),
                    "background_tolerance": 2e-8, "resolved_light_ring_count": count}
            report["samples"].append(item)
            if bg is not None:
                item.update(screening(bg, direct=scale == .06))
                print(f"sigma={scale:.4f}, A0={central:.8f}: T_N={item['fixed_charge_action']:.8g}, "
                      f"Q_trial={item['radial_energy']['direct_mass_second_variation']:.8g}", flush=True)
            else:
                print(f"sigma={scale:.4f}: {count} resolved light rings; optical screen omitted", flush=True)
            save(report, output)
        assert bg is not None
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DATA/"s06a044-background.npz", r=bg.sol.x, coefficients=bg.sol.c,
                            central=bg.central, scale=bg.scale,
                            ring_brackets=light_ring_brackets(solution, bg.central, bg.scale))
        with np.load(DATA/"s06a044-background.npz") as data:
            restored = Background(path=DATA/"s06a044-background.npz", central=float(data["central"]),
                                  scale=float(data["scale"]), ring_brackets=data["ring_brackets"])
        assert np.array_equal(restored.sol(bg.sol.x), bg.sol(bg.sol.x))
        assert restored.rs == bg.rs and restored.ru == bg.ru
        report["endpoint_sha256"] = hashlib.sha256((DATA/"s06a044-background.npz").read_bytes()).hexdigest()
        endpoint_checks(bg, report["samples"][-1], report, output)
        report["status"] = "completed_joint_screen"
        report["elapsed_seconds"] = perf_counter()-start
        save(report, output)
    except Exception as error:
        report["status"] = "stopped"
        report["error"] = repr(error)
        report["elapsed_seconds"] = perf_counter()-start
        save(report, output)
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA/"joint-screen.json")
    args = parser.parse_args()
    result = verification(args.output)
    print(json.dumps({"status": result["status"], "samples": len(result["samples"]),
                      "elapsed_seconds": result["elapsed_seconds"]}))
