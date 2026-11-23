"""Screen principal quadratic gravity and slow beat frequencies, not a flux proof."""

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import brentq
import sympy as sp
from sympy.physics.wigner import gaunt


ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'stable-light-rings'
sys.path.insert(0, str(OLD))
from finite_width_axial import OrbitSource
from trapped_axial import Background, gauss_mesh


def principal_checks():
    p, q, dp, dq, ddp, ddq = sp.symbols('p q dp dq ddp ddq', real=True)
    eta = sp.Matrix([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def transverse(x, y):
        return sp.diag(sp.zeros(2), sp.Matrix([[x, y], [y, -x]]))

    h, dh, ddh = transverse(p, q), transverse(dp, dq), transverse(ddp, ddq)
    inverse1, dinverse1 = -eta*h*eta, -eta*dh*eta

    def connection(inverse, derivative):
        return [[[sum(inverse[a, d] * ((derivative[d, c] if b == 0 else 0)
                       + (derivative[d, b] if c == 0 else 0)
                       - (derivative[b, c] if d == 0 else 0)) for d in range(4))/2
                  for c in range(4)] for b in range(4)] for a in range(4)]

    c1 = connection(eta, dh)
    dc1 = connection(eta, ddh)
    c2 = connection(inverse1, dh)
    dc2a, dc2b = connection(dinverse1, dh), connection(inverse1, ddh)
    dc2 = [[[dc2a[a][b][c] + dc2b[a][b][c] for c in range(4)]
             for b in range(4)] for a in range(4)]
    r1, r2 = sp.zeros(4), sp.zeros(4)
    for b in range(4):
        for c in range(4):
            r1[b, c] = sp.expand(dc1[0][b][c] - (sum(dc1[a][b][a] for a in range(4)) if c == 0 else 0))
            r2[b, c] = sp.expand(dc2[0][b][c] - (sum(dc2[a][b][a] for a in range(4)) if c == 0 else 0)
                                + sum(c1[a][b][c]*c1[d][a][d] - c1[a][b][d]*c1[d][a][c]
                                      for a in range(4) for d in range(4)))
    assert r1 == sp.zeros(4)
    expected = sp.zeros(4)
    expected[0, 0] = p*ddp + q*ddq + (dp**2 + dq**2)/2
    assert r2 == expected
    assert sp.trace(eta*r2) == 0

    omega, nu, cosine = sp.symbols('omega nu cosine', real=True)
    # The scalar products of the two unit spatial directions are left symbolic.
    plus = -(omega + nu)**2 + omega**2 + nu**2 + 2*omega*nu*cosine
    minus = -(omega - nu)**2 + omega**2 + nu**2 - 2*omega*nu*cosine
    assert sp.expand(plus + 2*omega*nu*(1-cosine)) == 0
    assert sp.expand(minus - 2*omega*nu*(1-cosine)) == 0
    return dict(null_sum_and_difference=True, linear_ricci_zero=True,
                quadratic_ricci_only_uu=True, quadratic_ricci_uu=str(expected[0, 0]),
                transverse_radiative_source_zero=True, scalar_curvature_zero_to_second_order=True,
                scope='Frozen principal gravitational symbol with no leading scalar wave. Not the complete curved-star quadratic source.')


def angular_checks():
    cases = []
    for ell in (2, 5, 10):
        for delta in (2, 4):
            for m in sorted({0, ell//2, ell}):
                other = ell + delta
                coefficient = sp.binomial(2*delta, delta) / 2**delta
                for j in range(1, delta+1):
                    coefficient *= sp.sqrt(sp.Rational((ell+j)**2-m*m,
                                                        (2*(ell+j)-1)*(2*(ell+j)+1)))
                integral = (-1)**m * sp.sqrt(4*sp.pi/(2*delta+1)) * gaunt(ell, other, delta, m, -m, 0)
                assert sp.simplify(integral - coefficient) == 0
                factor = sp.Rational(ell*(ell+1)+other*(other+1)-delta*(delta+1), 2)
                factor /= sp.sqrt(ell*(ell+1)*other*(other+1))
                vector_integral = sp.simplify(factor*integral)
                assert vector_integral > 0
                cases.append(dict(ell=ell, other=other, m=m, output_degree=delta,
                                  scalar_P_projection=str(sp.simplify(integral)),
                                  unit_axial_vector_projection=str(vector_integral)))
    return dict(gaunt_and_recurrence_checks=len(cases), cases=cases,
                scope='One normalized axial-vector angular contraction. It does not exclude cancellation in the full Einstein source.')


def constraint_checks():
    # A finite-dimensional nonlinear map with a fixed right inverse tests the chart jets.
    a, x, y, z, v1, v2, d = sp.symbols('a x y z v1 v2 d')
    constraint = z + x*x + 2*x*y + y*z + x**3
    leading = sp.Matrix([v1, v2, 0])
    free2 = sp.Matrix([d, 0, 0])
    hessian = sp.hessian(constraint, (x, y, z)).subs({x: 0, y: 0, z: 0})
    u2 = free2 + sp.Matrix([0, 0, -(leading.T*hessian*leading)[0]/2])
    u3 = sp.Matrix([0, 0, -(leading.T*hessian*u2)[0]-v1**3])
    state = a*leading + a*a*u2 + a**3*u3
    defect = sp.expand(constraint.subs(dict(zip((x, y, z), state)), simultaneous=True))
    assert all(sp.expand(defect).coeff(a, k) == 0 for k in (0, 1, 2, 3))
    return dict(fixed_complement_constraint_jets=True,
                scope='Algebraic chart identity; no right-inverse theorem for the stellar preparation is asserted.')


def transport_checks():
    t, wr, wa, Or, Oa, k, j = sp.symbols('t wr wa Or Oa k j', real=True)
    phase = k*wa-j*wr-(k*Oa-j*Or)*t
    transport = lambda expr: sp.diff(expr, t)+Or*sp.diff(expr, wr)+Oa*sp.diff(expr, wa)
    assert sp.simplify(transport(phase)) == 0
    assert sp.simplify(transport(sp.cos(phase))) == 0
    return dict(slow_angle_modulation_solves_transport=True,
                positivity_condition='nonnegative occupation times (1+eta*cos(phase)), abs(eta)<=1',
                scope='Conserved kinetic radiation candidate on fixed invariant tori; coherent Einstein-wave realization is open.')


class FrequencyScreen:
    def __init__(self):
        self.background = Background()
        self.max_points = 0
        self.max_with_gap_points = 0
        self.calls = 0

    def orbit(self, xi, subdivisions):
        bg = self.background
        orbit = OrbitSource(bg, float(xi), order=80)
        internal = bg.sol.x[(bg.sol.x > orbit.a) & (bg.sol.x < orbit.b)]
        breaks = 2/np.pi*np.arcsin(np.sqrt((internal-orbit.a)/(orbit.b-orbit.a)))
        knots = np.unique(np.r_[np.linspace(0, 1, subdivisions+1), breaks])
        t, weights = gauss_mesh(knots, 8)
        if len(t) > 4000:
            raise RuntimeError('Frequency diagnostic exceeds its 4000 point per-integral budget.')
        self.max_points = max(self.max_points, len(t))
        distance = (orbit.b-orbit.a)*np.minimum(np.sin(np.pi*t/2)**2, np.cos(np.pi*t/2)**2)
        with_gap_points = len(t) + 8*int(np.count_nonzero(distance < .001))
        if with_gap_points > 16000:
            raise RuntimeError('Quadrature and endpoint gap evaluation exceed 16000 radial points.')
        self.max_with_gap_points = max(self.max_with_gap_points, with_gap_points)
        self.calls += 1
        r, _, gap, ratio = orbit.mapped(t)
        weight = weights*ratio/bg.fields(r)['p']
        integral = float(sum(weight))
        action = float(weight@gap)/np.pi
        radial = np.pi/(np.sqrt(orbit.energy)*integral)
        azimuthal = float(weight@bg.fields(r)['U'])/(np.sqrt(orbit.energy)*integral)
        return dict(xi=float(xi), energy=float(orbit.energy), action_per_L=action,
                    Omega_r=radial, Omega_phi=azimuthal, ratio=azimuthal/radial,
                    delta_2_minus_8=2*azimuthal-8*radial,
                    delta_4_minus_17=4*azimuthal-17*radial,
                    turning_points=[orbit.a, orbit.b], points=len(t))

    def run(self):
        rows = []
        difference = 0.
        for xi in np.linspace(.998, .999, 10):
            coarse, fine = [self.orbit(xi, n) for n in (64, 128)]
            delta = max(abs(coarse[k]-fine[k]) for k in ('Omega_r', 'Omega_phi', 'delta_4_minus_17'))
            difference = max(difference, delta)
            rows.append(dict(**fine, quadrature_change=delta))
        roots = []
        for subdivisions in (64, 128):
            root = brentq(lambda xi: self.orbit(xi, subdivisions)['delta_4_minus_17'],
                          .998, .999, xtol=2e-11)
            roots.append(self.orbit(root, subdivisions))
        assert rows[0]['delta_4_minus_17'] < 0 < rows[-1]['delta_4_minus_17']
        assert difference < 2e-8 and abs(roots[0]['xi']-roots[1]['xi']) < 2e-8

        # Independent differentiation of the radial action at fixed frequency and L.
        center, step = .9985, 1e-5
        middle = self.orbit(center, 128)
        low, high = self.orbit(center-step, 128), self.orbit(center+step, 128)
        derivative = (high['action_per_L']-low['action_per_L'])/(high['energy']-low['energy'])
        omega_r_from_action = 1/(2*np.sqrt(middle['energy'])*derivative)
        omega_phi_from_action = np.sqrt(middle['energy'])-middle['action_per_L']*omega_r_from_action
        relative = max(abs(omega_r_from_action/middle['Omega_r']-1),
                       abs(omega_phi_from_action/middle['Omega_phi']-1))
        assert relative < 1e-5
        root = roots[-1]
        depth = self.background.Uu-self.background.Us
        # At zero beat, changing L by four changes e by -8e/L to first order.
        required_L = 8*root['energy']/(depth*(root['xi']-.998))
        return dict(status='floating_geodesic_frequency_screen', rows=rows,
                    ratio_17_over_4_roots=roots, maximum_quadrature_change=difference,
                    action_derivative_relative_check=relative,
                    local_carrier_scale_to_keep_pair_in_band=float(required_L),
                    maximum_points_per_integral=self.max_points, orbit_evaluations=self.calls,
                    maximum_points_including_gap_derivative=self.max_with_gap_points,
                    bvp_solves=0, mode_eigenvalues_computed=0,
                    scope='Represented-background geodesic frequencies, not interval bounds, discrete wave frequencies, or outgoing amplitudes.')


def main():
    if not __debug__:
        raise SystemExit('Run checks without python -O.')
    started = perf_counter()
    screen = FrequencyScreen()
    report = dict(status='quadratic_channel_screened', principal=principal_checks(),
                  angular=angular_checks(), preparation=constraint_checks(),
                  transport=transport_checks(), orbits=screen.run())
    paths = [OLD/'data/s08a06/background.npz', OLD/'finite_width_axial.py', OLD/'trapped_axial.py',
             ROOT.parents[1]/'literature/self-radiation/sources/1703.09069v3/main.tex',
             ROOT.parents[1]/'literature/self-radiation/arXiv-1703.09069v3.tex.gz']
    report['input_sha256'] = {str(p.relative_to(ROOT.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report['implementation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['seconds'] = perf_counter()-started
    (ROOT/'data/quadratic-channels.json').write_text(json.dumps(report, indent=2)+'\n', encoding='ascii')
    print(json.dumps(dict(status=report['status'], principal=report['principal'],
                          angular_checks=report['angular']['gaunt_and_recurrence_checks'],
                          root=report['orbits']['ratio_17_over_4_roots'][-1]['xi'],
                          quadrature_change=report['orbits']['maximum_quadrature_change'],
                          maximum_radial_points=screen.max_with_gap_points,
                          seconds=report['seconds']), indent=2))


if __name__ == '__main__':
    main()
