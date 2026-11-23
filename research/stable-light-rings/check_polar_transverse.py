"""Certify a positive matrix deformation using the dipole zero solution.

The proposed transverse scalar deformation is a binary Hermite polynomial
on the wall, a rational function in the core, and zero in the exterior.
Its inequality is verified on the exact enclosed background independently
of the numerical ODE used to propose it.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from time import perf_counter

if not __debug__:
    raise SystemExit("Do not disable interval verification with python -O.")

import numpy as np
import sympy as sp
from mpmath import iv
from scipy.interpolate import CubicHermiteSpline

from check_asymptotic_exterior import lower
from check_background_defect import field_equations
from check_barrier_certificate import Jet, enclosure, exact
from check_finite_static_inverse import upper
from check_polar_coercivity import check_inputs, physical_cell
from check_polar_factorization import certify_tail
from check_regular_center import Dual, integrands, leading, matter
from polar_transverse import deformation_proposal, transverse_coefficients
from trapped_axial import Background

ROOT = Path(__file__).parent
REFERENCE = ROOT/"data/solitonic-screen/polar-transverse-deformation.npz"


def scalar_coefficients(r, y):
    a, b, u, zeta = y
    A0, sigma, pi = iv.mpf(".06"), iv.mpf(".08"), iv.pi
    if isinstance(r, Jet):
        A0, sigma, pi = [Jet.coerce(v, r.order) for v in (A0, sigma, pi)]
    A, Ap, f = A0*a, A0*b, 1-2*u
    exponent = 2*zeta
    O2 = exponent.exp() if isinstance(exponent, Jet) else iv.exp(exponent)
    x = A*A/(sigma*sigma)
    v, Vs, Vss = (1-2*x)*(1-2*x), (1-2*x)*(1-6*x), (-8+24*x)/(sigma*sigma)
    V = A*A*v
    rho, pr = (O2*A*A+f*Ap*Ap+V)/2, (O2*A*A+f*Ap*Ap-V)/2
    n = (u/r+4*pi*r*pr)/f
    fp = 2*u/r-8*pi*r*rho
    d = 2/r+n+fp/(2*f)
    F = 4*pi*(2*O2*A*A-V)
    C = 4*n*Ap+2*A*(2*O2-Vs)/f
    vv = 4*n*n+8*pi*(V-4*O2*A*A)/f
    uu = (Vs+2*A*A*Vss-O2)/f+16*pi*Ap*Ap
    norm = n*n+4*pi*Ap*Ap
    cross = (n*(Vs-O2)*A-Ap*F)/f
    turn2 = 4*pi*cross*cross/(norm*norm)
    potential = 2/(r*r*f)+(4*pi*Ap*Ap*vv-8*pi*n*Ap*C+n*n*uu)/norm-3*turn2
    return potential, d


def core_deformation(r):
    return 11*r/(4*(1+r)), 11/(4*(1+r)*(1+r))


def hermite(r, left, right, values, slopes):
    h = exact(right-left)
    t = (r-exact(left))/h
    a, b = [exact(float(v)) for v in values]
    ap, bp = [exact(float(v)) for v in slopes]
    c2, c3 = 3*(b-a)-h*(2*ap+bp), 2*(a-b)+h*(ap+bp)
    s = t*(t*(t*c3+c2)+h*ap)+a
    derivative = t*(t*(3*c3)+2*c2)/h+ap
    return s, derivative


def algebra_checks(bg):
    alpha, turn, s, dalpha, dturn, ds, d = sp.symbols("alpha turn s dalpha dturn ds d")
    T = sp.Matrix([[alpha, turn], [turn, s]])
    J = sp.Matrix([[0, -turn], [turn, 0]])
    derivative = sp.Matrix([[dalpha, dturn], [dturn, ds]])+J*T-T*J
    Qperp = sp.symbols("Qperp")
    Q = sp.Matrix([[dalpha+alpha**2+d*alpha-turn**2, dturn+2*alpha*turn+d*turn],
                   [dturn+2*alpha*turn+d*turn, Qperp]])
    remainder = sp.simplify(Q-derivative-d*T-T*T)
    assert remainder == sp.diag(0, Qperp-3*turn**2-ds-d*s-s**2)
    x, h = sp.symbols("x h", real=True, nonzero=True)
    a, b, ap, bp = sp.symbols("a b ap bp")
    c2, c3 = 3*(b-a)-h*(2*ap+bp), 2*(a-b)+h*(ap+bp)
    cubic = a+x*(h*ap+x*(c2+x*c3))
    dcubic = sp.diff(cubic, x)/h
    assert cubic.subs(x, 0) == a and sp.expand(cubic.subs(x, 1)-b) == 0
    assert dcubic.subs(x, 0) == ap and sp.expand(dcubic.subs(x, 1)-bp) == 0
    assert sp.expand(dcubic-(ap+x*(2*c2+3*x*c3)/h)) == 0
    # A downward step contributes a positive interior boundary term.
    field, P = x*(2-x), 1+x*x
    difference = 0
    for left, right, deformation in ((0, 1, x), (1, 2, x-2)):
        potential = sp.diff(deformation, x)+sp.diff(P, x)*deformation/P+deformation**2
        integrand = P*(sp.diff(field, x)**2+potential*field**2
                      -(sp.diff(field, x)-deformation*field)**2)
        difference += sp.integrate(sp.expand(integrand), (x, left, right))
    assert difference == 4 == (P*field**2*2).subs(x, 1)
    y, yp, ypp, flux, fluxp, av, avp, density = [sp.Matrix(sp.symbols(name+"0:2"))
        for name in ("y", "yp", "ypp", "flux", "fluxp", "av", "avp", "density")]
    anisotropy, anisotropyp, k = sp.symbols("D Dp k")
    r0 = yp-flux-av*anisotropy
    r0p = ypp-fluxp-avp*anisotropy-av*anisotropyp
    r1 = fluxp+d*flux-Q*y-T*av*anisotropy-k*density
    forcing = av*anisotropyp+(avp+d*av+T*av)*anisotropy+k*density
    assert all(sp.expand(v) == 0 for v in
               ypp+d*yp-Q*y-forcing-((r1-T*r0)+r0p+d*r0+T*r0))
    rr = np.array([.1, 1., 15., 16., 17.3, 20., 40.])
    old = transverse_coefficients(bg, rr)
    differences = []
    for i, r in enumerate(rr):
        values = scalar_coefficients(exact(float(r)), [exact(float(v)) for v in bg.sol(r)])
        for actual, expected in zip(values, (old["potential"][i], old["d"][i])):
            error = abs(float(actual.mid)-expected)/(1+abs(expected))
            assert error < 3e-13
            differences.append(float(error))
    return {"rotating_frame_factorization": True, "hermite_value_and_slope_matching": True,
            "downward_jump_energy_identity": True, "symmetric_matrix_residual_identity": True,
            "coefficient_checks": len(differences),
            "maximum_scaled_coefficient_difference": max(differences)}


def central_bound(bg):
    r = enclosure(0, iv.mpf(".1"))
    radius = iv.mpf(".001")
    A0, sigma = iv.mpf(".06"), iv.mpf(".08")
    nominal = [iv.mpf(1), exact(float(bg.sol(0)[3]))]
    parameters = [v+enclosure(-iv.mpf("2.1e-8"), iv.mpf("2.1e-8")) for v in nominal]
    baseline = leading(nominal, A0, sigma, iv.pi)
    box = [v+enclosure(-radius, radius) for v in baseline]
    r2 = r*r
    factors = [iv.mpf(1)/2, iv.mpf(1)/3, iv.mpf(1)/3, iv.mpf(1)/2]
    image = integrands(r2, box, parameters, A0, sigma, iv.pi)
    image_radius = max(upper(abs(f*v-b)) for f, v, b in zip(factors, image, baseline))
    derivative = integrands(r2, [Dual(box[j], np.eye(4, dtype=int)[j]) for j in range(4)],
                           parameters, A0, sigma, iv.pi)
    contraction = max(upper(f*sum(map(abs, v.d), iv.mpf(0))) for f, v in zip(factors, derivative))
    assert exact(image_radius) < radius and exact(contraction) < 1
    p, q, mass, z = box
    a, zeta = parameters[0]+r2*p, parameters[1]+r2*z
    f, V, Vs, O2, rho, pr = matter(a, q, r2, mass, zeta, A0, sigma)
    A, bbar = A0*a, A0*q
    nbar = (mass+4*iv.pi*pr)/f
    assert lower(nbar) > 0
    dbar = nbar+(mass-4*iv.pi*rho)/f
    Vss = (-8+24*A*A/(sigma*sigma))/(sigma*sigma)
    C = 4*r2*nbar*bbar+2*A*(2*O2-Vs)/f
    vv = 4*r2*nbar*nbar+8*iv.pi*(V-4*O2*A*A)/f
    uu = (Vs+2*A*A*Vss-O2)/f+16*iv.pi*r2*bbar*bbar
    norm = nbar*nbar+4*iv.pi*bbar*bbar
    regular = (4*iv.pi*bbar*bbar*vv-8*iv.pi*nbar*bbar*C+nbar*nbar*uu)/norm
    cross = (nbar*(Vs-O2)*A-bbar*4*iv.pi*(2*O2*A*A-V))/f
    scaled_turn2 = 4*iv.pi*cross*cross/(norm*norm)
    s, ds = core_deformation(r)
    scaled_ds_term = iv.mpf(11)/4*(2*r2+r2*r2*dbar)/(1+r)
    residual = 2/f+r2*regular-3*scaled_turn2-r2*ds-scaled_ds_term-r2*s*s
    assert lower(residual) > 0
    return {"radius": ".1", "volterra_box_radius": ".001", "image_radius": image_radius,
            "contraction": contraction, "scaled_residual_lower": lower(residual)}


def energy_checks(bg, mesh, values, slopes):
    """Independent quadrature consistency; not an input to the interval proof."""
    spline = CubicHermiteSpline(mesh, values, slopes)
    edges = np.unique(np.r_[np.linspace(14, 15, 51), mesh, np.linspace(20, 21, 51)])
    results = []
    for order in (6, 10):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        width = np.diff(edges)/2
        r = ((edges[1:]+edges[:-1])[:, None]/2+width[:, None]*nodes).ravel()
        weights = (width[:, None]*weights).ravel()
        assert len(r) <= 16000
        c = transverse_coefficients(bg, r)
        s, ds = np.zeros_like(r), np.zeros_like(r)
        core, wall = r < 15, (r > 15) & (r < 20)
        s[core], ds[core] = 11*r[core]/(4*(1+r[core])), 11/(4*(1+r[core])**2)
        s[wall], ds[wall] = spline(r[wall]), spline(r[wall], 1)
        e, normal = c["e"], c["normal"]
        tensor = lambda x, y: np.einsum("ni,nj->nij", x, y)
        T = c["gamma"][:, None, None]*tensor(e, e)+s[:, None, None]*tensor(normal, normal)
        T += c["turn"][:, None, None]*(tensor(e, normal)+tensor(normal, e))
        amplitude = (r-14)**2*(21-r)**2/49
        derivative = 2*(r-14)*(21-r)*(35-2*r)/49
        y = np.column_stack((amplitude, amplitude*(r-17)/4))
        yp = np.column_stack((derivative, derivative*(r-17)/4+amplitude/4))
        residual = c["potential"]-ds-c["d"]*s-s*s
        shift = 4*c["W"]*np.sum(y*y, axis=1)
        original = np.sum(yp*yp, axis=1)+np.einsum("ni,nij,nj->n", y, c["Q1"], y)+shift
        deformed = np.sum((yp-np.einsum("nij,nj->ni", T, y))**2, axis=1)+shift
        deformed += residual*np.sum(normal*y, axis=1)**2
        at_jump = transverse_coefficients(bg, [20.])
        y_jump = 36/49*np.array([1., .75])
        jump = at_jump["P"][0]*values[-1]*float(at_jump["normal"][0]@y_jump)**2
        energy, reconstructed = float(weights@(c["P"]*original)), float(weights@(c["P"]*deformed)+jump)
        error = abs(energy-reconstructed)/(1+abs(energy))
        assert error < 2e-8
        results.append({"order": order, "points": len(r), "energy": energy,
                        "interface_term": jump, "relative_identity_error": error})
    return results


def tail_bound(digits):
    old = certify_tail(digits)
    assert exact(old["mass_pressure_interval"][0]) > iv.mpf("5.63")
    assert exact(old["mass_pressure_interval"][1]) < iv.mpf("5.67")
    invr = enclosure(0, iv.mpf(1)/40)
    mass = enclosure(iv.mpf("5.64"), iv.mpf("5.6600000001"))
    N = enclosure(iv.mpf("5.63"), iv.mpf("5.67"))
    f = 1-2*mass*invr
    z = iv.exp(enclosure(-iv.mpf("1e-11"), 0))
    omega = enclosure(iv.mpf(".139"), iv.mpf(".141"))
    O2 = omega*omega/(z*z*f)
    A = enclosure(0, iv.mpf("1e-8"))
    r2A = enclosure(0, 40**2*iv.mpf("1e-8"))
    t = enclosure(-iv.mpf("1.21"), -iv.mpf(".98"))
    x = A*A/iv.mpf(".08")**2
    v, Vs, Vss = (1-2*x)**2, (1-2*x)*(1-6*x), (-8+24*x)/iv.mpf(".08")**2
    n = N*invr*invr/f
    C = 4*n*A*t+2*A*(2*O2-Vs)/f
    vv = 4*n*n+8*iv.pi*A*A*(v-4*O2)/f
    uu = (Vs+2*A*A*Vss-O2)/f+16*iv.pi*A*A*t*t
    chi = iv.sqrt(4*iv.pi)*r2A*f*t/N
    turn = iv.sqrt(4*iv.pi)*r2A/N*((Vs-O2)-4*iv.pi*r2A*A*f*t*(2*O2-v)/N)/(1+chi*chi)
    potential_lower = lower((chi*chi*vv-2*chi*iv.sqrt(4*iv.pi)*C+uu)/(1+chi*chi)-3*turn*turn)
    assert exact(potential_lower) > iv.mpf(".97")
    return {"range": [40, "infinity"], "potential_lower": potential_lower,
            "r_squared_amplitude_upper": "0.000016", "uses_exact_exterior_family": True}


def verify_finite(bg, mesh, values, slopes):
    assert mesh[0] == 15 and mesh[-1] == 20 and values[0] == 165/64 and values[-1] > 0
    assert len(mesh) <= 1001 and values.shape == slopes.shape == mesh.shape
    assert np.all(np.isfinite(mesh)) and np.all(np.diff(mesh) > 0)
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(slopes))
    knots = [Fraction(float(v)) for v in bg.sol.x]
    nodes = [Fraction(float(v)) for v in mesh]
    edges = sorted({Fraction(".1"), Fraction(15), Fraction(20), Fraction(40)} |
                   {v for v in knots if Fraction(".1") < v < 40} | set(nodes))
    if len(edges) > 4000:
        raise RuntimeError("The finite transverse certificate exceeds 4000 partition points.")
    minima = [float("inf")]*3
    narrowed = 0
    def residual(r, y, zone, index):
        V, d = scalar_coefficients(r, y)
        if zone == 0:
            s, ds = core_deformation(r)
        elif zone == 1:
            s, ds = hermite(r, nodes[index], nodes[index+1], values[index:index+2], slopes[index:index+2])
        else:
            return V
        return V-ds-d*s-s*s
    for left, right in zip(edges[:-1], edges[1:]):
        middle = (left+right)/2
        piece = bisect_right(knots, middle)-1
        zone = 0 if right <= 15 else 1 if right <= 20 else 2
        index = min(max(bisect_right(nodes, middle)-1, 0), len(nodes)-2)
        r = enclosure(left, right)
        y = physical_cell(bg, piece, left, right)
        R = residual(r, y, zone, index)
        if lower(R) <= 0:
            center = exact(middle)
            yc = physical_cell(bg, piece, middle, middle)
            Rc = residual(center, yc, zone, index)
            flow = field_equations(r, y, iv.mpf(".06"), iv.mpf(".08"), iv.pi)
            rj = Jet(0, [r, iv.mpf(1)])
            yj = [Jet(0, [v, F]) for v, F in zip(y, flow)]
            derivative = residual(rj, yj, zone, index).c[1]
            bound = Rc+enclosure(-(right-left)/2, (right-left)/2)*derivative
            R = iv.mpf([max(R.a, bound.a), min(R.b, bound.b)])
            narrowed += 1
        assert lower(R) > 0, (float(left), float(right), zone, lower(R), upper(R))
        minima[zone] = min(minima[zone], lower(R))
    return {"partition_points": len(edges), "mean_value_narrowings": narrowed,
            "regions": [[.1, 15], [15, 20], [20, 40]], "residual_lower_bounds": minima,
            "downward_jump_at_20": float(values[-1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, default=30)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 25 <= args.digits <= 80:
        parser.error("Use 25 through 80 interval digits.")
    iv.dps = args.digits
    start = perf_counter()
    hashes = check_inputs()
    bg = Background()
    if args.build:
        if REFERENCE.exists():
            raise SystemExit("The saved deformation exists; verify it without --build.")
        proposal, diagnostic = deformation_proposal(bg)
        mesh = np.linspace(15, 20, 501)
        values = proposal.sol(mesh)[0]
        c = transverse_coefficients(bg, mesh)
        slopes = c["potential"]-c["d"]*values-values*values-.05
        np.savez(REFERENCE, mesh=mesh, values=values, slopes=slopes,
                 background_sha256=np.array(hashes["background.npz"]))
        print(json.dumps({"proposal": diagnostic}), flush=True)
    with np.load(REFERENCE) as data:
        assert str(data["background_sha256"]) == hashes["background.npz"]
        mesh, values, slopes = [data[key].copy() for key in ("mesh", "values", "slopes")]
    report = {"status": "all_polar_degrees_coercivity_verified", "digits": args.digits,
              "scope": "Exact original star; static polar degrees l>=2, not polar dynamical stability.",
              "input_sha256": hashes, "deformation_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
              "algebra": algebra_checks(bg), "center": central_bound(bg),
              "finite_region": verify_finite(bg, mesh, values, slopes), "tail": tail_bound(args.digits),
              "floating_energy_consistency": energy_checks(bg, mesh, values, slopes),
              "coercivity": "l*(l+1)-2", "elapsed_seconds": perf_counter()-start}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
