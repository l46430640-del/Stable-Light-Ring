"""Check dipole factorization, weak forcing, and an exact exterior bound.

Finite-background extrema and source norms are numerical diagnostics.
Only the explicitly stated exterior family is interval certified here.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

if not __debug__:
    raise SystemExit("Do not disable verification assertions with python -O.")

import numpy as np
import sympy as sp
from mpmath import iv
from scipy.optimize import minimize_scalar

from check_asymptotic_exterior import certify as certify_exterior, lower
from check_barrier_certificate import enclosure, exact
from check_finite_static_inverse import upper
from check_thin_wall_response import BACKGROUND, load_background, save
from polar_factorization import conditional_response_bound, dipole_coefficients, exterior_negative_trace
from polar_static import OrbitBandSource, SmoothOrbitSource, angular_parameters, polar_forcing
from trapped_axial import Background, gauss_mesh


def check_algebra():
    r, A, B, f, n, O, V, Vs, Vss = sp.symbols("r A B f n O V Vs Vss", nonzero=True)
    pi = sp.pi
    fp = (1-f)/r-4*pi*r*(f*B**2+O**2*A**2+V)
    d = 2/r+n+fp/(2*f)
    F = 4*pi*(2*O**2*A**2-V)
    np_ = F/f-n*n-(2/r+fp/(2*f))*n
    App = -d*B+(Vs-O**2)*A/f
    flow = {r: 1, A: B, B: App, f: fp, n: np_, O: -n*O, V: 2*A*B*Vs, Vs: 2*A*B*Vss}
    derivative = lambda h: sum(sp.diff(h, x)*xp for x, xp in flow.items())
    C = 4*n*B+2*A*(2*O**2-Vs)/f
    W, G = 1/(r*r*f), n-fp/(2*f)
    Q1 = sp.Matrix([[4*n*n+2*W+8*pi*(V-4*O**2*A*A)/f, 4*pi*C],
                    [C, 2*W+(Vs+2*A*A*Vss-O**2)/f+16*pi*B*B]])
    a, ap = sp.Matrix([n, B]), sp.Matrix([np_, App])
    residual = -ap.applyfunc(derivative)+(2*G-d)*ap+(derivative(G)+d*G-G*G)*a+Q1*a
    constraint_n = (1-f+4*pi*r*r*(f*B*B+O*O*A*A-V))/(2*r*f)
    reduced = [sp.factor(entry) for entry in residual]
    assert all(sp.cancel(entry.subs(n, constraint_n)) == 0 for entry in reduced)
    # The anisotropy source is an adjoint derivative of a*D.
    b = sp.Matrix([2*F/f-2*n*(1/r+n), -2*(1/r+n)*B+2*A*(Vs-O**2)/f])
    assert all(sp.cancel(entry) == 0 for entry in b-2*ap-(d-G)*a)
    S = sp.diag(np_/n-G, App/B-G)
    M = sp.Matrix([[-4*pi*C*B/n, 4*pi*C], [C, -C*n/B]])
    remainder = Q1-S.applyfunc(derivative)-d*S-S*S-M
    assert all(sp.cancel(entry.subs(n, constraint_n)) == 0 for entry in remainder)
    assert sp.cancel(M.det()) == 0
    assert all(sp.cancel(entry) == 0 for entry in M*a)
    assert sp.diag(1, 4*pi)*M == M.T*sp.diag(1, 4*pi)
    N, t = sp.symbols("N t", nonzero=True)
    trace = (r*r*f*C*(4*pi*B/n+n/B)).subs({B: A*t, n: N/(r*r*f)})
    formula = exterior_negative_trace(1/r, f, N, 4*pi*r*r*A*A, 4*pi*r**4*A*A, t, O*O, Vs)
    assert sp.cancel(trace-formula) == 0
    # Pure dipole gauge: R=1/z and T'=-R/(r^2 f) preserve g_rA=0.
    assert sp.cancel(-G-fp/(2*f)+n) == 0
    return {"dipole_gauge_zero": True, "factorization": True,
            "rank_one_remainder": True, "derivative_free_weak_source": True,
            "amplitude_free_exterior_formula": True}


def certify_tail(digits):
    exterior = certify_exterior(digits)
    R, amplitude = iv.mpf(40), iv.mpf("1e-8")
    s = enclosure(0, 1/R)
    mass = enclosure(iv.mpf("5.64"), iv.mpf("5.6600000001"))
    f = 1-2*mass*s
    z = iv.exp(enclosure(-iv.mpf("1e-11"), 0))
    omega = enclosure(iv.mpf(".139"), iv.mpf(".141"))
    omega2 = omega**2/(z*z*f)
    t = enclosure(-iv.mpf("1.21"), -iv.mpf(".98"))
    ratio = enclosure(0, amplitude**2/iv.mpf(".08")**2)
    v, vs = (1-2*ratio)**2, 1-8*ratio+12*ratio**2
    # All r^j A^2, j<=4, decrease for r>=40 by the existing .98 decay bound.
    d2, d3, d4 = [enclosure(0, 4*iv.pi*R**j*amplitude**2) for j in (2, 3, 4)]
    N = mass+d3*(omega2+f*t*t-v)/2
    assert exact(lower(N)) > iv.mpf("5.63")
    trace = exterior_negative_trace(s, f, N, d2, d4, t, omega2, vs)
    assert exact(lower(trace)) > iv.mpf("8.805")
    assert exact(upper(trace)) < iv.mpf("15.600")
    assert exact(upper(trace)) < 17
    return {"scope": "Exact original exterior family, r>=40 and 0<A(40)<=1e-8; no interior certificate.",
            "parameters": exterior["parameters"],
            "exterior_existence_contraction": exterior["verified_decimal_bounds"]["contraction"],
            "mass_pressure_interval": [lower(N), upper(N)],
            "negative_trace_interval": [lower(trace), upper(trace)],
            "verified_bound": "17", "coercivity_coefficient": "l*(l+1)-19",
            "arithmetic": f"{digits}-digit intervals over the entire exterior half-line"}


def trial(r, left, right, compact=False):
    x = (r-left)/(right-left)
    y = np.column_stack((1+.2*x+.1*x*x, .3-.1*x+.02*x*x))
    yp = np.column_stack((.2+.2*x, -.1+.04*x))/(right-left)
    if compact:
        h, hp = x*x*(1-x)**2, 2*x*(1-x)*(1-2*x)/(right-left)
        y, yp = h[:, None]*y, hp[:, None]*y+h[:, None]*yp
    return y, yp


def check_energy(bg, order, compact):
    left, right = .5*bg.rs, bg.radius-1
    r, weights = gauss_mesh(np.linspace(left, right, 161), order)
    assert len(r) <= 4000
    c = dipole_coefficients(bg, r)
    y, yp = trial(r, left, right, compact)
    old = weights@(c["P"]*(np.sum(yp*yp, axis=1)+np.einsum("ni,nij,nj->n", y, c["Q2"], y)))
    deformed = yp-c["S"]*y
    new = weights@(c["P"]*(np.sum(deformed**2, axis=1)+4*c["W"]*np.sum(y*y, axis=1)
                          +np.einsum("ni,nij,nj->n", y, c["M"], y)))
    ends = np.array([left, right])
    ce = dipole_coefficients(bg, ends)
    ye, _ = trial(ends, left, right, compact)
    boundary = ce["P"]*np.sum(ce["S"]*ye*ye, axis=1)
    error = abs(old-new-boundary[1]+boundary[0])/(1+abs(old))
    assert error < 2e-7, error
    return {"quadrature_order": order, "quadrature_points": len(r), "compact": compact, "energy": float(old),
            "square_and_remainder": float(new), "boundary": float(boundary[1]-boundary[0]),
            "scaled_identity_error": float(error)}


def source_diagnostic(source, order):
    bg = source.background
    knots = np.unique(np.r_[np.linspace(source.a, source.b, 101), source.knots])
    r, weights = gauss_mesh(knots, order)
    assert len(r) <= 4000
    c = dipole_coefficients(bg, r)
    rho, _, D, Dp = source.fields(r)
    g, a, ap = c["g"], c["a"], c["ap"]
    y, yp = trial(r, source.a, source.b)
    Dy = yp-c["S"]*y
    Z = a*Dp[:, None]+(ap+c["d"][:, None]*a+c["S"]*a)*D[:, None]
    for degree in (2, 8, 12):
        _, k, gain = angular_parameters(degree)
        expected = gain*Z
        expected[:, 0] += k*8*np.pi*rho/g["f"]
        forcing = polar_forcing(source, r, degree)
        actual = np.column_stack((forcing[1], np.sqrt(4*np.pi)*bg.central*forcing[3]))
        assert np.max(abs(actual-expected)/(1+abs(actual))) < 1e-13
    direct = weights@(c["P"]*np.sum(Z*y, axis=1))
    weak = -weights@(c["P"]*D*np.sum(a*Dy, axis=1))
    error = abs(direct-weak)/(1+abs(direct))
    # Differentiate the saved polynomial, independently of the background ODE.
    derivative = bg.sol(r, 1)
    Ar, Br, fr, Or = bg.central*derivative[0], bg.central*derivative[1], -2*derivative[2], g["Omega"]*derivative[3]
    prr = (g["Omega"]*Or*g["A"]**2+g["Omega"]**2*g["A"]*Ar
           +fr*g["Ap"]**2/2+g["f"]*g["Ap"]*Br-g["A"]*Ar*g["Vs"])
    nr = (derivative[2]/r-g["u"]/r**2+4*np.pi*g["pr"]+4*np.pi*r*prr-g["nup"]*fr)/g["f"]
    actual_ap = np.column_stack((nr, np.sqrt(4*np.pi)*Br))
    actual_d = 2/r-derivative[3]+fr/(2*g["f"])
    defect = weights@(c["P"]*D*np.sum((ap-actual_ap+(c["d"]-actual_d)[:, None]*a)*y, axis=1))
    corrected_error = abs(direct-weak-defect)/(1+abs(direct))
    assert error < 1e-8, error
    assert corrected_error < 1e-9, (type(source).__name__, bg.scale, order, direct, weak, defect, corrected_error)
    Rnorm2 = weights@(c["P"]*(8*np.pi*rho/g["f"])**2/c["W"])
    Anorm2 = weights@(c["P"]*np.sum(a*a, axis=1)*D*D)
    return {"quadrature_order": order, "quadrature_points": len(r), "direct_source_pairing": float(direct),
            "weak_source_pairing": float(weak), "scaled_identity_error": float(error),
            "background_defect_contribution": float(defect),
            "defect_corrected_error": float(corrected_error),
            "density_norm": float(np.sqrt(Rnorm2)), "anisotropy_norm": float(np.sqrt(Anorm2))}


def check_bound():
    # Cauchy--Schwarz is saturated when the two positive norm components align.
    L, B, R, A = 72, 52, 4.3, .47
    mu, k = L-2-B, 35/128
    e, y = conditional_response_bound(8, B, R, A)
    dual = np.array([R/np.sqrt(mu), 8*A/(L-2)])
    q = k*dual
    assert abs(q@q-k*(dual@q)) < 1e-14
    assert abs(e-np.linalg.norm(q))+abs(y-e/np.sqrt(mu)) < 1e-14
    assert conditional_response_bound(np.int64(80), B, R, A) == conditional_response_bound(80, B, R, A)
    rejected = 0
    for args in [(1, B, R, A), (3, B, R, A), (2.0, B, R, A),
                 (8, 70, R, A), (8, -1, R, A), (8, B, -1, A),
                 (8, B, R, np.inf), (8, np.nan, R, A)]:
        try:
            conditional_response_bound(*args)
        except ValueError:
            rejected += 1
    assert rejected == 8
    return {"sharp_two_component_cauchy_bound": True, "invalid_inputs_rejected": rejected}


def profile(bg, count):
    r = np.linspace(.002, bg.radius, count)
    c = dipole_coefficients(bg, r)
    i = int(np.argmax(c["negative_bound"]))
    result = minimize_scalar(lambda x: -dipole_coefficients(bg, [x])["negative_bound"][0],
                             bounds=(r[max(0, i-1)], r[min(count-1, i+1)]), method="bounded")
    assert result.success
    raw = c["Q2"]-6*c["W"][:, None, None]*np.eye(2)[None, :, :]
    raw_bound = np.maximum(-np.linalg.eigvalsh(raw)[:, 0]/c["W"], 0)
    g = c["g"]
    N = r*r*g["f"]*g["nup"]
    direct_trace = exterior_negative_trace(1/r, g["f"], N, 4*np.pi*r*r*g["A"]**2,
                                          4*np.pi*r**4*g["A"]**2, g["Ap"]/g["A"],
                                          g["Omega"]**2, g["Vs"])
    assert np.max(abs(direct_trace+np.trace(c["M"], axis1=1, axis2=2)/c["W"])/(1+abs(direct_trace))) < 1e-12
    null_error = np.max(abs(np.einsum("nij,nj->ni", c["M"], c["a"])))
    assert null_error < 1e-12
    return {"points": count, "range": [.002, float(bg.radius)],
            "deformed_maximum": float(-result.fun), "maximum_radius": float(result.x),
            "sampled_undeformed_maximum": float(np.max(raw_bound)),
            "deformed_endpoint_value": float(c["negative_bound"][-1]),
            "analytic_asymptotic_value": float(max(2*bg.mass*(1-2*bg.omega**2)/np.sqrt(1-bg.omega**2), 0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, default=30)
    parser.add_argument("--certificate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 25 <= args.digits <= 80:
        raise ValueError("Use 25 to 80 interval digits.")
    started = perf_counter()
    report = {"algebra": check_algebra(), "bound_formula": check_bound(), "tail_certificate": certify_tail(args.digits)}
    if not args.certificate_only:
        report["finite_background_diagnostics"] = []
        for bg, path in [(Background(), Path(__file__).parent/"data/s08a06/background.npz"),
                         (load_background(), BACKGROUND)]:
            entry = {"background": str(path.relative_to(Path(__file__).parent)),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "profiles": [profile(bg, count) for count in (1201, 2401)],
                     "energy_identities": [check_energy(bg, order, compact)
                                           for compact in (False, True) for order in (8, 12)],
                     "sources": [[source_diagnostic(source, order) for order in (24, 32)]
                                 for source in (SmoothOrbitSource(bg), OrbitBandSource(bg))]}
            assert abs(entry["profiles"][0]["deformed_maximum"]-entry["profiles"][1]["deformed_maximum"]) < 1e-5
            for pair in entry["sources"]:
                assert max(abs(pair[0][key]-pair[1][key]) for key in ("density_norm", "anisotropy_norm")) < 1e-9
            report["finite_background_diagnostics"].append(entry)
    report["elapsed_seconds"] = perf_counter()-started
    if args.output:
        save(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
