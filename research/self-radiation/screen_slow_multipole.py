"""Reciprocal static-multipole screen for the slow orbital modulation.

All stellar results are floating diagnostics, not interval certificates or
physical radiation estimates. Existing background and forced fields are read-only.
"""

import hashlib
import json
from math import comb, factorial
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from mpmath import mp
from numpy.polynomial import Legendre
from numpy.polynomial.legendre import legvander
from scipy.optimize import brentq
import sympy as sp

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'stable-light-rings'
sys.path.insert(0, str(OLD))
from finite_width_axial import OrbitSource
from polar_residual_budget import RepresentedResponse
from polar_static import OrbitBandSource, PolarResponse, angular_constraint, coefficients
from trapped_axial import Background, gauss_mesh
from check_quadratic_channels import FrequencyScreen


class ZeroSource:
    def __init__(self, background):
        self.background = background
        self.a, self.b = 15.62, 16.53
        self.knots = np.array([self.a, background.rs, self.b])

    def fields(self, radius):
        return np.zeros((4, np.atleast_1d(radius).size))


class TidalResponse(PolarResponse):
    def __init__(self, background, **kwargs):
        self.tidal_scale = background.rs
        super().__init__(ZeroSource(background), **kwargs)

    def growing(self, r, derivative=0):
        ell, mass = self.degree, self.background.mass
        x = np.asarray(r)/mass-1
        p = Legendre.basis(ell)
        leading = comb(2*ell, ell)*ell*(ell-1)/2**ell
        factor = (mass/self.tidal_scale)**ell/leading
        if derivative == 0:
            return factor*(x*x-1)*p.deriv(2)(x)
        return factor*(2*x*p.deriv(2)(x)+(x*x-1)*p.deriv(3)(x))/mass

    def boundary(self, left, right):
        result = super().boundary(left, right)
        result[2] -= self.growing(self.outer, 1)-self.metric_slope*self.growing(self.outer)
        return result


def decaying_vacuum(mass, r, ell):
    with mp.workdps(60):
        M, rr = mp.mpf(float(mass)), mp.mpf(float(r))
        x = rr/M-1
        previous = mp.log((x+1)/(x-1))/2
        current = x*previous-1
        for j in range(1, ell):
            previous, current = current, ((2*j+1)*x*current-j*previous)/(j+1)
        derivative = ell*(x*current-previous)/(x*x-1)
        associated = ell*(ell+1)*current-2*x*derivative
        leading = mp.mpf(2)**ell*factorial(ell)**2/factorial(2*ell+1)
        return float(associated/(leading*(ell+1)*(ell+2)*M**(ell+1)))


def phase_quadrature(orbit, tide, subdivisions, order=8):
    bg = orbit.background
    radial_knots = np.unique(np.r_[bg.sol.x, tide.solution.x])
    radial_knots = radial_knots[(radial_knots > orbit.a) & (radial_knots < orbit.b)]
    mapped_knots = 2/np.pi*np.arcsin(np.sqrt((radial_knots-orbit.a)/(orbit.b-orbit.a)))
    knots = np.unique(np.r_[np.linspace(0, 1, subdivisions+1), mapped_knots])
    x, q = np.polynomial.legendre.leggauss(order)
    half = np.diff(knots)/2
    t = ((knots[:-1]+knots[1:])[:, None]/2+half[:, None]*x).ravel()
    distance = (orbit.b-orbit.a)*np.minimum(np.sin(np.pi*t/2)**2, np.cos(np.pi*t/2)**2)
    total_points = len(t)+8*int(np.count_nonzero(distance < .001))
    if total_points > 16000:
        raise RuntimeError('Orbit and endpoint gap quadrature exceed 16000 radial points.')
    r, _, gap, ratio = orbit.mapped(t)
    g = bg.fields(r)
    dt = (np.sqrt(orbit.energy)*ratio/g['p']).reshape(-1, order)
    dphi = (g['U']*ratio/g['p']).reshape(-1, order)
    # Integrate the Gauss interpolant from each cell's left endpoint to each node.
    integrals = np.column_stack([(Legendre.basis(j).integ()(x)
                                 - Legendre.basis(j).integ()(-1)) for j in range(order)])
    transform = ((2*np.arange(order)+1)[:, None]/2)*legvander(x, order-1).T*q
    partial = integrals@transform

    def accumulated(values):
        cell_integrals = half*(values@q)
        prefix = np.r_[0., np.cumsum(cell_integrals[:-1])]
        return (prefix[:, None]+half[:, None]*(values@partial.T)).ravel(), float(sum(cell_integrals))

    time, half_period = accumulated(dt)
    angle, half_angle = accumulated(dphi)
    weights = (half[:, None]*q*dt).ravel()/half_period
    assert abs(sum(weights)-1) < 1e-13
    return r, time/half_period, angle, half_period, half_angle, weights, gap, total_points


def orbit_overlap(tide, xi, k=4, j=17, subdivisions=64, order=8):
    if tide.degree != k:
        raise ValueError('Use the tide with the same angular degree as the modulation.')
    bg = tide.background
    orbit = OrbitSource(bg, xi)
    r, scaled_time, phi, half_period, half_angle, weights, gap, points = phase_quadrature(orbit, tide, subdivisions, order)
    v, vp, _, u, up, _, w, _, _ = tide.physical_jet(r)
    phase = k*phi+(j*np.pi-k*half_angle)*scaled_time
    U_over_e = bg.fields(r)['U']/orbit.energy
    complete = 2*v-U_over_e*w
    density_only = v
    angular_w = angular_constraint(bg, r, np.array([v, vp, u/bg.central, up/bg.central]), degree=k)+v
    reconstructed = 2*v-U_over_e*angular_w
    cosine = np.cos(phase)
    overlap = float(weights@(complete*cosine))
    angular_factor = comb(2*k, k)/(2**(2*k))*(np.sqrt(3)/2)**k
    return dict(xi=xi, k=k, j=j, overlap=overlap,
                density_only_overlap=float(weights@(density_only*cosine)),
                angular_constraint_overlap=float(weights@(reconstructed*cosine)),
                absolute_integral=float(weights@abs(complete*cosine)),
                peak_metric=float(np.max(abs(np.r_[v, w]))),
                Omega_r=np.pi/half_period, Omega_phi=half_angle/half_period,
                beat_frequency=(k*half_angle-j*np.pi)/half_period,
                static_delta_multipole_diagnostic_per_energy_eta=-angular_factor*overlap*tide.tidal_scale**k,
                angular_factor=angular_factor, quadrature_points_including_gap=points,
                subdivisions=subdivisions, quadrature_order=order)


def old_source_reciprocity(tide, order=12):
    bg, ell = tide.background, tide.degree
    path = OLD/f'data/solitonic-screen/polar-residual-fields/degree-{ell:03d}-tol-2e-07.npz'
    with np.load(path) as data:
        factor = float(data['source_factor'])
    source = OrbitBandSource(bg, (.998, .999))
    source.factor = factor
    response = RepresentedResponse(source, ell, 2e-7)
    # The reference forced field enters only through its exterior coefficient.
    # Its interior interpolation knots are not breaks of this work integrand.
    knots = np.unique(np.r_[source.knots, tide.solution.x, bg.sol.x])
    knots = knots[(knots >= source.a) & (knots <= source.b)]
    available = min(len(knots)-1, 16000//12-(len(knots)-1))
    if available < 0:
        raise RuntimeError('Reference-source base partition exceeds the quadrature budget.')
    selected = np.argsort(np.diff(knots))[-available:] if available else []
    midpoints = ((knots[:-1]+knots[1:])/2)[selected]
    knots = np.sort(np.r_[knots, midpoints])
    r, weights = gauss_mesh(knots, order)
    if len(r) > 16000:
        raise RuntimeError('Reference-source reciprocity integral exceeds 16000 points.')
    v, _, _, _, _, _, w, _, _ = tide.physical_jet(r)
    rho, pressure, _, _ = source.fields(r)
    g = bg.fields(r)
    radial_work = weights@(r*r*g['alpha']/np.sqrt(g['f'])*response.funk*(w*(rho-pressure)-2*v*rho))
    predicted = 4*np.pi/(2*ell+1)*radial_work*tide.tidal_scale**ell
    extracted = response.sol(response.outer)[0]/decaying_vacuum(bg.mass, response.outer, ell)
    return dict(predicted=float(predicted), extracted=float(extracted),
                relative_difference=float(abs(predicted/extracted-1)), quadrature_points=len(r), order=order,
                input_path=str(path.relative_to(ROOT.parents[1])), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def algebra_checks():
    rho, pressure, f, d, jr, jrp, w, wp, a = sp.symbols('rho pressure f d jr jrp w wp a')
    lam, r, v = sp.symbols('lam r v', nonzero=True)
    D = -8*sp.pi*r*r*(rho-pressure+2*f*(jrp+d*jr))/(lam-2)
    # Integration by parts maps -w*(j'+d*j) to w'*j with weight P'=d*P.
    integrand = (lam-2)*w*D/(2*r*r*f)+8*sp.pi*v*rho/f+16*sp.pi*a*jr
    remainder = sp.expand(integrand+4*sp.pi*(w*(rho-pressure)-2*v*rho)/f)
    assert sp.simplify(remainder-(-8*sp.pi*w*(jrp+d*jr)+16*sp.pi*a*jr)) == 0
    assert sp.simplify((8*sp.pi*wp*jr+16*sp.pi*a*jr).subs(wp, -2*a)) == 0
    theta, inclination = sp.symbols('theta inclination', real=True)
    for k in (2, 4):
        average = sp.integrate(sp.legendre(k, sp.sin(inclination)*sp.cos(theta))*sp.cos(k*theta), (theta, 0, 2*sp.pi))/(2*sp.pi)
        expected = sp.binomial(2*k, k)*sp.sin(inclination)**k/2**(2*k)
        assert sp.trigsimp(average-expected) == 0
    A, Ap, Om2, V, Vs, Vss, n, fp = sp.symbols('A Ap Om2 V Vs Vss n fp')
    vp, u, up, Dp = sp.symbols('vp u up Dp')
    D = sp.Symbol('D')
    F = 4*sp.pi*(2*Om2*A*A-V)
    damping = 2/r+n+fp/(2*f)
    npp = F/f-n*n-(2/r+fp/(2*f))*n
    App = -damping*Ap+(Vs-Om2)*A/f
    C = 4*n*Ap+2*A*(2*Om2-Vs)/f
    Av = 4*n*n+lam/(r*r*f)+8*sp.pi*(V-4*Om2*A*A)/f
    Au = (lam/r**2+Vs+2*A*A*Vss-Om2)/f+16*sp.pi*Ap*Ap
    Jv = (2*F/f-2*n*(1/r+n))*D+n*Dp+8*sp.pi*rho/f+16*sp.pi*n*jr
    Ju = Ap*(Dp-2*(1/r+n)*D)+2*A*(Vs-Om2)*D/f+16*sp.pi*Ap*jr
    vpp, upp = -damping*vp+Av*v+4*sp.pi*C*u+Jv, -damping*up+Au*u+C*v+Ju
    cp = -vp-2*n*v+(1/r+n)*D-8*sp.pi*Ap*u-8*sp.pi*jr
    b, bp = D-v, Dp-vp
    tt = f*(vpp+(2*n+2/r+fp/(2*f))*vp+n*(2*cp-bp))+2*(v-b)*F-lam*v/r**2
    tt -= 8*sp.pi*((2*Om2-Vs)*A*u-V*v+rho)
    scalar = f*(upp+damping*up)+(Om2-Vs-2*A*A*Vss-lam/r**2)*u
    scalar += f*(vp-bp+2*cp)*Ap-2*b*(Vs-Om2)*A-2*v*Om2*A
    mixed = -vp-cp+(v+b)/r+n*(b-v)-8*sp.pi*(Ap*u+jr)
    assert all(sp.expand(value) == 0 for value in (tt, scalar, mixed))
    # Independent angular-trace reconstruction of the homogeneous tidal field.
    homogeneous = {D: 0, Dp: 0, rho: 0, jr: 0}
    cp0 = cp.subs(homogeneous)
    cpp0 = -vpp.subs(homogeneous)-2*npp*v-2*n*vp-8*sp.pi*(App*u+Ap*up)
    c0 = (f*r*r*cpp0+f*r*(4+r*n+r*fp/(2*f))*cp0+2*r*f*vp
          +2*(1-4*sp.pi*r*r*V)*v+8*sp.pi*r*r*Vs*A*u)/(lam-2)
    aa = n*v+4*sp.pi*Ap*u
    ap = npp*v+n*vp+4*sp.pi*(App*u+Ap*up)
    B = 2*F*v/f+8*sp.pi*A*(Vs-Om2)*u/f-2*(1/r+n)*aa
    assert sp.simplify(B-ap-damping*aa-(lam-2)*(c0+v)/(2*r*r*f)) == 0
    # The null Hamiltonian work retains both radial and tangential pressures.
    Ue = sp.Symbol('Ue')
    hamiltonian = v-(-v)*(1-Ue)-(w-v)*Ue
    assert sp.expand(hamiltonian-(2*v-Ue*w)) == 0
    # Covariant divergence of the mixed stress, independently in coordinates.
    theta = sp.Symbol('theta', real=True)
    tau, pp, alpha = sp.symbols('tau pp alpha')
    Gamma = sp.MutableDenseNDimArray.zeros(4, 4, 4)
    for aidx, bidx, cidx, value in (
        (0, 0, 1, n), (1, 0, 0, f*alpha**2*n), (1, 1, 1, -fp/(2*f)),
        (1, 2, 2, -f*r), (1, 3, 3, -f*r*sp.sin(theta)**2),
        (2, 1, 2, 1/r), (3, 1, 3, 1/r), (2, 3, 3, -sp.sin(theta)*sp.cos(theta)),
        (3, 2, 3, sp.cot(theta))):
        Gamma[aidx, bidx, cidx] = Gamma[aidx, cidx, bidx] = value

    def derivative(value, coordinate):
        if coordinate == 1:
            return sp.diff(value, r)+sp.diff(value, f)*fp+sp.diff(value, jr)*jrp+sp.diff(value, pressure)*pp
        return sp.diff(value, theta) if coordinate == 2 else 0

    for ell in (2, 4):
        eigenvalue = ell*(ell+1)
        Y = sp.legendre(ell, sp.cos(theta))
        Yp, Ypp = sp.diff(Y, theta), sp.diff(Y, theta, 2)
        q = (rho-pressure)/2
        T = sp.diag(-rho*Y, pressure*Y, q*Y+tau*(Ypp+eigenvalue*Y/2),
                    q*Y+tau*(sp.cot(theta)*Yp+eigenvalue*Y/2))
        T[1, 2], T[2, 1] = f*jr*Yp, jr*Yp/r**2
        expected = [0, (pp+n*(rho+pressure)+(3*pressure-rho)/r-eigenvalue*jr/r**2)*Y,
                    (f*(jrp+damping*jr)+q-(eigenvalue-2)*tau/2)*Yp, 0]
        assert sp.trigsimp(sp.trace(T)) == 0
        for nu in range(4):
            divergence = sum(derivative(T[mu, nu], mu) for mu in range(4))
            divergence += sum(Gamma[mu, mu, index]*T[index, nu]-Gamma[index, mu, nu]*T[mu, index]
                              for mu in range(4) for index in range(4))
            assert sp.trigsimp(sp.expand(divergence-expected[nu])) == 0
    return dict(mixed_stress_cancellation=True, plane_angular_projection=True,
                general_forcing_tt_scalar_mixed=True, homogeneous_angular_identity=True,
                null_hamiltonian_work=True, coordinate_stress_trace_and_divergence_checks=10)


def main():
    if not __debug__:
        raise SystemExit('Run this screen without python -O.')
    started = perf_counter()
    bg = Background()
    frequency_record = json.loads((ROOT/'data/quadratic-channels.json').read_text())
    roots = {4: frequency_record['orbits']['ratio_17_over_4_roots'][-1]['xi']}
    frequency_screen = FrequencyScreen()
    roots2 = []
    for n in (64, 128):
        root = brentq(lambda x: frequency_screen.orbit(x, n)['delta_2_minus_8'], .995, .998, xtol=2e-11)
        roots2.append(frequency_screen.orbit(root, n))
    assert abs(roots2[0]['xi']-roots2[1]['xi']) < 2e-8
    roots[2] = roots2[-1]['xi']
    report = dict(status='slow_static_multipole_screen_in_progress', algebra=algebra_checks(),
                  quadrupole_roots=roots2, trials=[])
    path = ROOT/'data/slow-static-multipole.json'
    attempts = [(4, 1e-6, .002, 40.), (4, 2e-7, .002, 40.), (4, 2e-7, .001, 38.),
                (2, 1e-6, .002, 40.), (2, 2e-7, .002, 40.), (2, 2e-9, .002, 40.), (2, 2e-9, .001, 38.)]
    for degree, tolerance, inner, outer in attempts:
        try:
            tide = TidalResponse(bg, degree=degree, tolerance=tolerance, inner=inner, outer=outer, max_nodes=4000)
        except RuntimeError as error:
            report.update(status='stopped_at_tidal_node_budget', stopped_attempt=dict(degree=degree, tolerance=tolerance, inner=inner, outer=outer, error=str(error)))
            path.write_text(json.dumps(report, indent=2)+'\n', encoding='ascii')
            raise
        row = dict(degree=degree, tolerance=tolerance, inner=inner, outer=outer, nodes=len(tide.solution.x),
                   residuals=tide.residuals(), overlap=[orbit_overlap(tide, roots[degree], k=degree,
                       j={2: 8, 4: 17}[degree], subdivisions=n) for n in (64, 128)])
        if outer == 40.:
            row['reference_source_reciprocity'] = [old_source_reciprocity(tide, order) for order in (8, 12)]
            assert max(item['relative_difference'] for item in row['reference_source_reciprocity']) < 2e-6
        filename = f'slow-tide-l{degree}-tol-{tolerance:.0e}-R{outer:g}.npz'
        np.savez_compressed(ROOT/'data'/filename, knots=tide.solution.x, state=tide.solution.y,
                            slope=tide.solution.sol(tide.solution.x, 1), degree=degree,
                            tidal_scale=tide.tidal_scale, tolerance=tolerance, inner=inner, outer=outer)
        row['field_path'] = filename
        row['field_sha256'] = hashlib.sha256((ROOT/'data'/filename).read_bytes()).hexdigest()
        report['trials'].append(row)
        path.write_text(json.dumps(report, indent=2)+'\n', encoding='ascii')
        print(json.dumps(dict(degree=degree, tolerance=tolerance, outer=outer, nodes=row['nodes'], overlap=row['overlap'][-1],
                             reciprocity=row.get('reference_source_reciprocity')), indent=2), flush=True)
    inputs = [OLD/'data/s08a06/background.npz', OLD/'polar_static.py', OLD/'finite_width_axial.py',
              OLD/'trapped_axial.py', OLD/'polar_residual_budget.py', ROOT/'check_quadratic_channels.py',
              ROOT/'data/quadratic-channels.json']
    report.update(status='floating_slow_static_multipole_screen', seconds=perf_counter()-started,
                  root_orbit_evaluations=frequency_screen.calls,
                  root_maximum_points_including_gap=frequency_screen.max_with_gap_points,
                  input_sha256={str(p.relative_to(ROOT.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Static reciprocal response on represented background and finite endpoint conditions. No rigorous nonzero certificate, wave realization or outgoing-news estimate.')
    path.write_text(json.dumps(report, indent=2)+'\n', encoding='ascii')


if __name__ == '__main__':
    main()
