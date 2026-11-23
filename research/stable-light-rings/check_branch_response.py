"""Explore the charge projection across a solitonic-star turning point.

All branch and response values are floating-point diagnostics, not existence
or sign certificates. Every boundary solve retains the 4000-node limit.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq
import sympy as sp

from boson_star import boundary, potential, solve_background
from concentrated_axial import ConcentratedResponse, bottom_action_variation
from static_transversality import BranchTangent, background_charge, bottom_barrier_action
from trapped_axial import Background


AMPLITUDES = (.06, .06005, .06010, .06015, .06020, .06030, .06050, .061, .0612, .062)


def light_ring_brackets(solution, central, scale):
    r = solution.x
    a, b, u, zeta = solution.y
    V, _ = potential(central*a, scale)
    pressure = (np.exp(2*zeta)*(central*a)**2+(1-2*u)*(central*b)**2-V)/2
    H = 3*u+4*np.pi*r*r*pressure-1
    crossings = np.flatnonzero(H[:-1]*H[1:] < 0)
    if len(crossings) != 2:
        raise ValueError(f"Continuation has {len(crossings)} resolved light rings, not the required pair.")
    return [(float(r[i]), float(r[i+1])) for i in crossings]


class Branch:
    def __init__(self, background_tolerance=2e-8, response_tolerance=2e-6, radius=40.):
        self.background_tolerance = background_tolerance
        self.response_tolerance = response_tolerance
        self.radius = radius
        self.seed = Background()
        self.cache = {}

    def equilibrium(self, central):
        central = float(central)
        if central not in self.cache:
            if self.cache:
                nearest = min(self.cache, key=lambda a: abs(a-central))
                previous = self.cache[nearest][0]
            else:
                previous = SimpleNamespace(sol=self.seed.sol)
            solution = solve_background(central=central, radius=self.radius, tolerance=self.background_tolerance,
                                        max_nodes=4000, previous=previous)
            if not solution.success:
                raise RuntimeError(f"Branch solve stopped at A0={central} within 4000 nodes: {solution.message}")
            brackets = light_ring_brackets(solution, central, .08)
            bg = Background(solution=solution, central=central, ring_brackets=brackets)
            assert solution.x.size <= 4000
            assert np.min(solution.y[0]) > 0 and np.min(1-2*solution.y[2]) > .2
            assert 0 < bg.omega < 1
            assert np.max(np.abs(boundary(solution.y[:, 0], solution.y[:, -1], self.radius))) < 1e-10
            g_s, g_u = bg.fields(bg.rs), bg.fields(bg.ru)
            crossing = [8*np.pi*r*r*g["Omega"]**2*g["A"]**2-1
                        for r, g in ((bg.rs, g_s), (bg.ru, g_u))]
            assert crossing[0] > 0 and crossing[1] < 0
            tangent = BranchTangent(bg, tolerance=self.response_tolerance)
            self.cache[central] = (solution, bg, tangent)
        return self.cache[central]

    def charge_slope(self, central):
        _, bg, tangent = self.equilibrium(central)
        return float(tangent.charge_response/bg.central)

    def response(self, central, direct=False):
        solution, bg, tangent = self.equilibrium(central)
        free = ConcentratedResponse(bg, tolerance=self.response_tolerance, fixed_charge=False)
        Np = float(free.state(bg.radius)[4])
        Tp, outer = free.limiting_action()
        fine = free.limiting_action(240)[0]
        dS = bottom_action_variation(bg, tangent.metric)[0]/bg.central
        dN = tangent.charge_response/bg.central
        compensation = -Np*dS/dN if dN != 0 else float("nan")
        report = {"central_amplitude": float(central), "mass": float(bg.mass), "charge": background_charge(bg),
                  "background_tolerance": self.background_tolerance, "response_tolerance": self.response_tolerance,
                  "omega": float(bg.omega), "light_rings": [float(bg.rs), float(bg.ru)],
                  "relative_well_depth": float((bg.Uu-bg.Us)/bg.Us), "bottom_action": bottom_barrier_action(bg),
                  "outer_turning_point": outer, "dN_dA0": float(dN), "dS0_dA0": float(dS),
                  "auxiliary_charge": Np, "auxiliary_action": Tp, "charge_adjustment_action": float(compensation),
                  "fixed_charge_action_from_projection": float(Tp+compensation),
                  "action_quadrature_change": abs(fine-Tp),
                  "background_nodes": int(solution.x.size), "tangent_nodes": int(tangent.solution.x.size),
                  "auxiliary_response_radial_nodes": int(2*free.solution.x.size),
                  "tangent_first_law_error": tangent.diagnostics()["first_law_error"],
                  "source_first_law_error": float(abs(bg.radius*free.state(bg.radius)[2]-1-bg.omega*Np))}
        assert report["action_quadrature_change"] < 2e-6
        assert report["tangent_first_law_error"] < 2e-6
        assert report["source_first_law_error"] < 2e-6
        assert report["auxiliary_response_radial_nodes"] <= 4000
        if direct:
            fixed = ConcentratedResponse(bg, tolerance=self.response_tolerance)
            actual = fixed.limiting_action()[0]
            radii = np.unique(np.r_[np.linspace(0, bg.radius, 301), bg.rs])
            coefficient = -Np/tangent.charge_response
            discrepancy = fixed.state(radii)-free.state(radii)-coefficient*tangent.solution.sol(radii)
            error = float(np.max(np.abs(discrepancy)))
            scaling = np.array([1., 1., 1., 1., 1/report["charge"]])
            scaled_error = float(np.max(np.abs(discrepancy)*scaling[:, None]))
            report["direct_fixed_charge_response"] = {
                "action": actual, "projection_action_difference": abs(actual-Tp-compensation),
                "field_projection_difference": error, "scaled_field_projection_difference": scaled_error,
                "added_mass": float(bg.radius*fixed.state(bg.radius)[2]),
                "final_charge_response": float(fixed.state(bg.radius)[4]),
                "radial_nodes": int(2*fixed.solution.x.size)}
            assert abs(actual-Tp-compensation) < 2e-5
            assert scaled_error < 3e-5
            assert abs(bg.radius*fixed.state(bg.radius)[2]-1) < 2e-6
            assert abs(fixed.state(bg.radius)[4]) < 1e-9
            assert 2*fixed.solution.x.size <= 4000
        return report


def check_algebra():
    t, a, b, c, d, q, p, h = sp.symbols("t a b c d q p h", nonzero=True)
    Nprime, Np, Sprime, Tp = a*t, b+c*t, d+q*t, p+h*t
    response = Tp-Np*Sprime/Nprime
    residue = -b*d/a
    assert sp.simplify(sp.limit(t*response, t, 0)-residue) == 0
    shift = sp.Function("g")(t)
    shifted = Tp+shift*Sprime-(Np+shift*Nprime)*Sprime/Nprime
    assert sp.simplify(shifted-response) == 0
    cancellation = response.subs(b, 0)
    assert sp.simplify(sp.limit(t*cancellation, t, 0)) == 0
    print("PASS: simple-pole residue, auxiliary-response invariance and vanishing-coupling exception", flush=True)


def verification():
    if not __debug__:
        raise RuntimeError("Run without -O: branch diagnostics require assertions.")
    start = perf_counter()
    check_algebra()
    branch = Branch()
    records = []
    for central in AMPLITUDES:
        report = branch.response(central, direct=central in (.06, .0603, .061, .062))
        records.append(report)
        print(f"A0={central:.8f}: N'={report['dN_dA0']:.7g}, "
              f"T_N={report['fixed_charge_action_from_projection']:.9g}", flush=True)
    assert records[3]["dN_dA0"] > 0 > records[4]["dN_dA0"]
    assert records[3]["fixed_charge_action_from_projection"] > 0 > records[4]["fixed_charge_action_from_projection"]
    assert records[7]["fixed_charge_action_from_projection"] < 0 < records[8]["fixed_charge_action_from_projection"]
    root = brentq(branch.charge_slope, .06015, .0602, xtol=2e-11)
    refined = Branch(background_tolerance=2e-9, response_tolerance=2e-7)
    critical = brentq(refined.charge_slope, root-1e-7, root+1e-7, xtol=2e-12)
    # No division by the tiny charge slope is needed at the turning point.
    _, bg, tangent = refined.equilibrium(critical)
    critical_charge_slope = float(tangent.charge_response/critical)
    free = ConcentratedResponse(bg, tolerance=refined.response_tolerance, fixed_charge=False)
    Np = float(free.state(bg.radius)[4])
    Sprime = bottom_action_variation(bg, tangent.metric)[0]/critical
    curvature = [(refined.charge_slope(critical+h)-refined.charge_slope(critical-h))/(2*h)
                 for h in (1e-6, 5e-7)]
    residue = -Np*Sprime/curvature[-1]
    assert abs(root-critical) < 2e-9
    assert all(v < 0 for v in curvature) and Np < 0 < Sprime
    assert abs(curvature[0]/curvature[1]-1) < 1e-4
    near = []
    for distance in (1e-5, 5e-6, 2.5e-6):
        for sign in (-1, 1):
            central = critical+sign*distance
            report = refined.response(central)
            scaled = (central-critical)*report["fixed_charge_action_from_projection"]
            near.append({"central_amplitude": central, "signed_distance": sign*distance,
                         "fixed_charge_action": report["fixed_charge_action_from_projection"],
                         "scaled_response": scaled, "residue_relative_difference": abs(scaled/residue-1),
                         "radial_nodes": [report["background_nodes"], report["tangent_nodes"],
                                          report["auxiliary_response_radial_nodes"]]})
    assert max(v["residue_relative_difference"] for v in near[-2:]) < .02
    for side in (0, 1):
        assert near[side+4]["residue_relative_difference"] < near[side+2]["residue_relative_difference"]
        assert near[side+2]["residue_relative_difference"] < near[side]["residue_relative_difference"]
    refinements = []
    # The tighter A0=0.062 background reached the node cap in an earlier run.
    # Keep its converged coarse result; do not retry that refinement.
    for index in (0, 5):
        fine = refined.response(records[index]["central_amplitude"], direct=True)
        fine["action_tolerance_change"] = abs(fine["fixed_charge_action_from_projection"]-
                                               records[index]["fixed_charge_action_from_projection"])
        assert fine["action_tolerance_change"] < 3e-5
        refinements.append(fine)
    # Independent neighboring equilibria check both tangent observables.
    difference_check = []
    center = .0603
    _, model, tangent = refined.equilibrium(center)
    predicted = [tangent.charge_response/center, bottom_action_variation(model, tangent.metric)[0]/center]
    for step in (1e-6, 5e-7):
        values = []
        for sign in (-1, 1):
            _, model, _ = refined.equilibrium(center+sign*step)
            values.append([background_charge(model), bottom_barrier_action(model)])
        derivative = (np.array(values[1])-values[0])/(2*step)
        error = np.abs(derivative-predicted)
        assert error[0] < .01 and error[1] < .003
        difference_check.append({"step": step, "dN_dA0": float(derivative[0]), "dS0_dA0": float(derivative[1]),
                                 "derivative_errors": error.tolist()})
    assert np.all(np.array(difference_check[1]["derivative_errors"]) <
                  .3*np.array(difference_check[0]["derivative_errors"]))
    alternate = Branch(background_tolerance=2e-9, response_tolerance=2e-7, radius=30.)
    exterior_check = alternate.response(.0603, direct=True)
    exterior_change = abs(exterior_check["fixed_charge_action_from_projection"]-
                          refinements[1]["fixed_charge_action_from_projection"])
    assert exterior_change < 3e-5
    all_nodes = [len(sol.x) for source in (branch, refined, alternate) for sol, _, _ in source.cache.values()]
    all_nodes += [len(tan.solution.x) for source in (branch, refined, alternate) for _, _, tan in source.cache.values()]
    return {"status": "passed_branch_numerical_checks",
            "scope": "Floating-point continuation at fixed sigma0=0.08. Negative concentrated monopole response occurs after a numerical charge maximum; neither exact-background signs nor radial dynamical stability are certified.",
            "seed_background_sha256": hashlib.sha256((Path(__file__).parent/"data/s08a06/background.npz").read_bytes()).hexdigest(),
            "max_nodes_per_boundary_solve": 4000, "maximum_background_or_tangent_nodes": max(all_nodes),
            "sampled_branch": records, "turning_point": {
                "coarse_central_amplitude": root, "refined_central_amplitude": critical,
                "tolerance_shift": abs(root-critical), "mass": float(bg.mass),
                "charge_slope_at_refined_point": critical_charge_slope,
                "charge_curvatures": curvature, "auxiliary_charge": Np, "bottom_action_slope": float(Sprime),
                "action_response_residue": float(residue)},
            "near_turning_point": near, "tolerance_refinements": refinements,
            "equilibrium_difference_checks": difference_check, "outer_radius_check": exterior_check,
            "outer_radius_action_change": exterior_change, "elapsed_seconds": perf_counter()-start}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verification()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
