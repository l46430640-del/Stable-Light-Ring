"""Enclose center and infinity residuals of global polar approximations.

The finite response is defined by exact shared Hermite nodal data. Attach
a regular power at the center and an exact Schwarzschild metric mode plus
an exponential scalar at infinity. Flux jumps are bounded in the energy
dual. Interior residuals and optical constants are separate inputs.
"""

import argparse
import hashlib
import json
from math import acosh, ceil, log
from pathlib import Path
from time import perf_counter

if not __debug__:
    raise SystemExit("Run endpoint certificates without python -O.")

import numpy as np
import sympy as sp
from mpmath import iv

from check_asymptotic_exterior import lower
from check_barrier_certificate import enclosure, exact
from check_finite_static_inverse import upper
from check_polar_coercivity import check_inputs, physical_cell
from check_polar_factorization import certify_tail
from check_polar_transverse import central_bound
from check_regular_center import leading, matter
from trapped_axial import Background

ROOT = Path(__file__).parent
CACHE = ROOT/"data/solitonic-screen/polar-residual-fields"


def norm(v):
    return iv.sqrt(sum((x**2 for x in v), iv.mpf(0)))


def algebra_checks():
    r, M, delta, pressure, V = sp.symbols("r M delta pressure V")
    f0, f = 1-2*M/r, 1-2*(M+delta)/r
    n = ((M+delta)/r**2+4*sp.pi*r*pressure)/f
    assert sp.factor(n-M/(r*r*f0)-delta/(r*r*f*f0)-4*sp.pi*r*pressure/f) == 0
    d = 2/r+n+((M+delta)/r**2-4*sp.pi*r*(pressure+V))/f
    assert sp.factor(d-(2/r+2*M/(r*r*f0))-2*delta/(r*r*f*f0)+4*sp.pi*r*V/f) == 0
    x, q = sp.symbols("x q", positive=True)
    f = 1-2*x
    F = lambda p: p*p-p/f-q*(q-1)/f-4*x*x/(f*f)
    assert sp.factor(F(q)+2*x*q*q/f+4*x*x/(f*f)) == 0
    assert sp.factor(F(q/f)+x*sp.diff(q/f, x)-2*x*(q*q-2*x)/(f*f)) == 0
    s, R, rate = sp.symbols("s R rate", positive=True)
    assert sp.simplify(sp.integrate((R+s)**2*sp.exp(-rate*s), (s, 0, sp.oo))
                       -(R*R/rate+2*R/rate**2+2/rate**3)) == 0
    return {"metric_coefficient_differences": True, "vacuum_slope_barriers": True,
            "exponential_tail_moment": True}


def central_coefficients(bg, radius):
    A0, sigma, pi = iv.mpf(".06"), iv.mpf(".08"), iv.pi
    nominal = [iv.mpf(1), exact(float(bg.sol(0)[3]))]
    parameters = [v+enclosure(-iv.mpf("2.1e-8"), iv.mpf("2.1e-8")) for v in nominal]
    box = [v+enclosure(-iv.mpf(".001"), iv.mpf(".001")) for v in leading(nominal, A0, sigma, pi)]
    p, q, mass, z = box
    r2 = radius**2
    a, zeta = parameters[0]+r2*p, parameters[1]+r2*z
    f, V, Vs, O2, rho, pr = matter(a, q, r2, mass, zeta, A0, sigma)
    A, bbar = A0*a, A0*q
    nbar = (mass+4*pi*pr)/f
    dbar = nbar+(mass-4*pi*rho)/f
    Vss = (-8+24*A*A/(sigma*sigma))/(sigma*sigma)
    cross = iv.sqrt(4*pi)*(4*r2*nbar*bbar+2*A*(2*O2-Vs)/f)
    vv = 4*r2*nbar*nbar+8*pi*(V-4*O2*A*A)/f
    uu = (Vs+2*A*A*Vss-O2)/f+16*pi*r2*bbar*bbar
    alpha = enclosure(iv.mpf(".139"), iv.mpf(".141"))*iv.exp(-zeta)
    return dict(A=A, bbar=bbar, nbar=nbar, dbar=dbar, mass=mass, f=f,
                V=V, Vs=Vs, O2=O2, rho=rho, vv=vv, uu=uu, cross=cross, alpha=alpha)


def trace_bound(bg, inner):
    r = enclosure(inner, 2*inner)
    c = central_coefficients(bg, r)
    n, Ap, f = r*c["nbar"], r*c["bbar"], c["f"]
    d = 2/r+r*c["dbar"]
    np_ = 4*iv.pi*(2*c["O2"]*c["A"]**2-c["V"])/f-d*n
    App = (c["Vs"]-c["O2"])*c["A"]/f-d*Ap
    G = n-r*(c["mass"]-4*iv.pi*c["rho"])/f
    denominator = n*n+4*iv.pi*Ap*Ap
    gamma = (n*np_+4*iv.pi*Ap*App)/denominator-G
    turn = iv.sqrt(4*iv.pi)*(n*App-Ap*np_)/denominator
    Tbound = upper(abs(gamma)+abs(turn)+11*r/(4*(1+r)))
    Pmin = lower(r*r*c["alpha"]*iv.sqrt(f))
    PWmin = lower(c["alpha"]/iv.sqrt(f))
    assert Pmin > 0 and PWmin > 0
    return exact(Tbound), exact(Pmin), exact(PWmin)


def vacuum_slope_interval(mass, radius, degree, digits):
    saved = iv.dps
    work = max(digits, ceil(2*(degree+1)*acosh(float(radius)/float(mass)-1)/log(10))+80)
    if work > 800:
        raise RuntimeError("Vacuum recurrence exceeds the 800-digit work limit.")
    iv.dps = work
    try:
        M, R = exact(float(mass)), exact(float(radius))
        x = R/M-1
        previous = iv.ln((x+1)/(x-1))/2
        current = x*previous-1
        for ell in range(1, degree):
            previous, current = current, ((2*ell+1)*x*current-ell*previous)/(ell+1)
        L = degree*(degree+1)
        derivative = degree*(x*current-previous)/(x*x-1)
        Q = L*current-2*x*derivative
        assert lower(Q) > 0
        Qp = (L-2)*derivative-2*x*Q/(x*x-1)
        result = Qp/(M*Q)
        assert upper(result.b-result.a) < 1e-30
        return result, work
    finally:
        iv.dps = saved


def tail_constants(bg, digits):
    old = certify_tail(digits)
    assert old["verified_bound"] == "17"
    R, amplitude = iv.mpf(40), iv.mpf("1e-8")
    M = exact(float(bg.mass))
    state = physical_cell(bg, len(bg.sol.x)-2, 40, 40)
    mass_error = upper(abs(R*state[2]-M)+14000*amplitude**2)
    assert exact(mass_error) < iv.mpf("1.16e-7")
    fmin = 1-2*iv.mpf("5.6600000001")/R
    assert 1/fmin < iv.mpf("1.4")
    assert M < iv.mpf("5.66") and iv.mpf("5.67")/fmin < 8
    f = enclosure(fmin, 1)
    A2 = enclosure(0, amplitude**2)
    x = A2/iv.mpf(".08")**2
    v, Vs, Vss = (1-2*x)**2, (1-2*x)*(1-6*x), (-8+24*x)/iv.mpf(".08")**2
    omega = enclosure(iv.mpf(".139"), iv.mpf(".141"))
    z = iv.exp(enclosure(-iv.mpf("1e-11"), 0))
    O2 = omega**2/(z*z*f)
    assert (O2+iv.mpf("1.21")**2+v)/2 < iv.mpf("1.25")
    assert abs(v-4*O2) < 1 and abs(2*O2-Vs) < 1 and abs(2*O2-v) < 1 and abs(Vs-O2) < 1
    scalar = (Vs+2*A2*Vss-O2)/f+16*iv.pi*A2*iv.mpf("1.21")**2
    assert scalar > 0 and scalar < iv.mpf("1.5")
    assert iv.sqrt(4*iv.pi)*(4*8/R**2*iv.mpf("1.21")+2*iv.mpf("1.4")) < 11
    assert 2/R+16/R**2 < iv.mpf(".061")
    gamma_extra = 4*iv.pi*amplitude**2*(R**2/iv.mpf("5.63")
                     +iv.mpf("1.21")*iv.mpf("1.4")*R**4/iv.mpf("5.63")**2)
    turn = iv.sqrt(4*iv.pi)*R**2*amplitude/iv.mpf("5.63")*(1
                     +4*iv.pi*R**2*amplitude**2*iv.mpf("1.21")/iv.mpf("5.63"))
    assert 2/R+16/R**2+gamma_extra+turn < iv.mpf(".07")
    assert R**2*fmin*iv.exp(-iv.mpf("1e-11")) > 1100
    return {"mass_error_upper": mass_error, "deformation_norm_upper": ".07",
            "P_lower": "1100", "PW_lower": ".99", "coupling_per_amplitude_upper": "11"}


def certificate(bg, degree, path, central, traces, tail, digits):
    with np.load(path) as data:
        assert int(data["degree"]) == degree
        knots, state, slope = [data[k].copy() for k in ("knots", "state", "slope")]
        assert len(knots) <= 4000 and np.all(np.diff(knots) > 0)
        assert state.shape == slope.shape == (5, len(knots))
        assert np.all(np.isfinite(state)) and np.all(np.isfinite(slope))
        assert str(data["background_sha256"]) == hashlib.sha256((ROOT/"data/s08a06/background.npz").read_bytes()).hexdigest()
    inner, R = exact(float(knots[0])), exact(float(knots[-1]))
    assert R == 40 and inner == exact(.002)
    scale = iv.sqrt(4*iv.pi)*iv.mpf(".06")
    left, right = [[exact(float(v)) for v in state[:, index]] for index in (0, -1)]
    yl, yr = [left[0], scale*left[2]], [right[0], scale*right[2]]
    pl, pr = [left[1], scale*left[3]], [right[1], scale*right[3]]
    result = field_endpoint_bounds(bg, degree, yl, yr, pl, pr, central, traces, tail, digits)
    result["field_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["center_angular_L1_upper"] = upper(abs(left[4])+2*norm(
        [central["nbar"], iv.sqrt(4*iv.pi)*central["bbar"]])*norm(yl)*inner**2/(degree+2))
    return result


def field_endpoint_bounds(bg, degree, yl, yr, pl, pr, central, traces, tail, digits):
    """Homogeneous endpoint defects for arbitrary physical two-component fields."""
    inner, R = exact(.002), iv.mpf(40)
    L, q, mu = degree*(degree+1), degree+1, degree*(degree+1)-2
    B = degree*central["dbar"]-2*L*central["mass"]/central["f"]
    C = max(upper(abs(B-central[k])+abs(central["cross"])) for k in ("vv", "uu"))
    center_norm = exact(C)*norm(yl)*inner**iv.mpf("2.5")/iv.sqrt(mu*(2*degree+5))
    T, Pmin, PWmin = traces
    Cinner = (2/inner+4*inner*T*T)/(mu*PWmin)+4*inner/Pmin
    inner_jump = norm([pl[j]-degree*yl[j]/inner for j in range(2)])
    inner_functional = inner**2*iv.sqrt(Cinner)*inner_jump

    beta, aR, delta = iv.mpf(".98"), iv.mpf("1e-8"), exact(tail["mass_error_upper"])
    I2 = lambda rate: R*R/rate+2*R/(rate*rate)+2/(rate**3)
    Nv = delta*(4*L+6*q+4)*abs(yr[0])/iv.sqrt(R*(2*q+1))
    Nv += (26*q+72)*R*R*aR*aR*abs(yr[0])/(2*iv.sqrt(beta))
    Nv += 11*aR*abs(yr[1])*R*iv.sqrt(I2(2*(beta+1)))
    Nu = 11*aR*abs(yr[0])*R*R/iv.sqrt(2*beta)
    Nu += (3+iv.mpf(L)/1000)*abs(yr[1])*R*iv.sqrt(I2(iv.mpf(2)))
    exterior_norm = norm([Nv, Nu])/iv.sqrt(mu)
    vacuum, precision = vacuum_slope_interval(bg.mass, 40, degree, digits)
    outer_jump = norm([vacuum*yr[0]-pr[0], -(1+1/R)*yr[1]-pr[1]])
    Couter = (2+4*iv.mpf(".07")**2)/(mu*iv.mpf(".99"))+iv.mpf(4)/1100
    outer_functional = R*R*iv.sqrt(Couter)*outer_jump
    return {"degree": degree, "center_residual_upper": upper(center_norm),
            "center_coefficient_norm_upper": C, "inner_flux_jump_upper": upper(inner_jump),
            "inner_interface_dual_upper": upper(inner_functional),
            "exterior_residual_upper": upper(exterior_norm), "outer_flux_jump_upper": upper(outer_jump),
            "outer_interface_dual_upper": upper(outer_functional), "vacuum_work_digits": precision,
            "endpoint_energy_dual_upper": upper(center_norm+inner_functional+exterior_norm+outer_functional)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ROOT/"data/solitonic-screen/polar-endpoint-budget.json")
    args = parser.parse_args()
    if not 25 <= args.digits <= 80:
        parser.error("Use 25 through 80 interval digits.")
    iv.dps = args.digits
    start = perf_counter()
    bg = Background()
    hashes = check_inputs()
    center = central_bound(bg)
    central = central_coefficients(bg, enclosure(0, exact(.002)))
    traces = trace_bound(bg, exact(.002))
    tail = tail_constants(bg, args.digits)
    report = {"scope": "Rigorous energy-dual bounds for center, infinity and endpoint flux jumps of the specified global Hermite approximations; interior and optical errors not certified.",
              "digits": args.digits, "input_sha256": hashes, "center_box": center, "tail_constants": tail,
              "algebra": algebra_checks(),
              "rows": [], "status": "endpoint_residuals_verified"}
    for degree in range(2, 257, 2):
        tolerance = 2e-7 if degree <= 16 else 2e-6
        path = CACHE/f"degree-{degree:03d}-tol-{tolerance:.0e}.npz"
        report["rows"].append(certificate(bg, degree, path, central, traces, tail, args.digits))
    report["elapsed_seconds"] = perf_counter()-start
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "degrees": len(report["rows"]),
                      "tail_constants": tail, "quadrupole": report["rows"][0],
                      "maximum_work_digits": max(r["vacuum_work_digits"] for r in report["rows"]),
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
