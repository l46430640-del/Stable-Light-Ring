"""Validate a resonant azimuth orbit by a continuous boundary inverse.

A shared cubic state and an invertible shared cubic matrix define the
reference problem exactly. Coercivity of the tidal field is not used here.
The exact background enclosure is inherited from the stellar certificate.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from mpmath import iv
from scipy.integrate import solve_ivp
import sympy as sp

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'stable-light-rings'
sys.path.insert(0, str(OLD))
from check_asymptotic_exterior import lower
from check_background_defect import field_equations
from check_barrier_certificate import Jet, exact, enclosure
from check_finite_static_inverse import upper, interval_norm
from check_polar_coercivity import physical_cell, check_inputs
from check_polar_transverse import hermite
from check_relative_optics import RelativePotential
from trapped_axial import Background
from finite_width_axial import OrbitSource

REFERENCE = ROOT/'data/azimuth-orbit-reference.npz'
SCALE = (1., .2, .001)
SCALE_EXACT = ('1', '.2', '.001')

if not __debug__:
    raise SystemExit('Run certificates without python -O.')


def physical_flow(r, z, q, y):
    """Flow and its state Jacobian, with stellar radial derivatives from the ODE."""
    isjet = isinstance(r, Jet)
    pi = Jet(iv.pi, order=r.order) if isjet else np.pi
    a0, sigma = (iv.mpf('.06'), iv.mpf('.08')) if isjet else (.06, .08)
    A, Ap, f = y[0]*a0, y[1]*a0, 1-2*y[2]
    O2 = (2*y[3]).exp() if isjet else np.exp(2*y[3])
    x = A*A/(sigma*sigma)
    V, Vs = A*A*(1-2*x)*(1-2*x), (1-2*x)*(1-6*x)
    rho, pressure = (O2*A*A+f*Ap*Ap+V)/2, (O2*A*A+f*Ap*Ap-V)/2
    n = (y[2]/r+4*pi*r*pressure)/f
    fp = 2*y[2]/r-8*pi*r*rho
    d = 2/r+n+fp/(2*f)
    np_ = 4*pi*(2*O2*A*A-V)/f-d*n
    App = (Vs-O2)*A/f-d*Ap
    rhop = -n*O2*A*A+O2*A*Ap+fp*Ap*Ap/2+f*Ap*App+Vs*A*Ap
    fpp = -4*y[2]/(r*r)-8*pi*r*rhop
    L, Lp = 2*n-2/r, 2*np_+2/(r*r)
    K, Kp, Kpp = r*r*f, 2*r*f+r*r*fp, 2*f+4*r*fp+r*r*fpp
    G = (Kp*q-K*(1+q)*L)/8
    Gr = (Kpp*q-Kp*(1+q)*L-K*(1+q)*Lp)/8
    flow = [z, G, -(1+q)*L*z]
    jacobian = [[0, 1, 0], [Gr, 0, (Kp-K*L)/8],
                [-(1+q)*Lp*z, -(1+q)*L, -L*z]]
    dz = [[0, 0, 0], [0, 0, 0], [-(1+q)*Lp, 0, -L]]
    dq = [[0, 0, 0], [(Kpp-Kp*L-K*Lp)/8, 0, 0], [-Lp*z, -L, 0]]
    return flow, jacobian, dz, dq


def build_reference(bg, count):
    assert count in (128, 256, 512)
    scale = np.array(SCALE)
    initial_r = OrbitSource(bg, .6113144108037043).a
    evaluations = 0
    trials = []
    def rhs(t, values):
        nonlocal evaluations
        evaluations += 1
        if evaluations > 16000:
            raise RuntimeError('Reference construction exceeded 16000 RHS evaluations.')
        r, z, q = values[:3]*scale
        flow, jac, _, _ = physical_flow(r, z, q, bg.sol(r))
        flow = 5*np.pi*np.asarray(flow)/scale
        jac = 5*np.pi*np.asarray(jac)*scale[None, :]/scale[:, None]
        return np.r_[flow, (jac@values[3:].reshape(3, 3)).ravel()]
    for _ in range(3):
        initial = np.r_[initial_r, 0., 0., np.eye(3).ravel()]
        solution = solve_ivp(rhs, (0., 1.), initial, method='DOP853', max_step=1/128,
                             rtol=2e-10, atol=2e-12, dense_output=True)
        assert solution.success and len(solution.t) < 4000
        end = solution.y[:, -1]
        D = end[3:].reshape(3, 3)[1, 0]
        assert abs(D) > .1
        trials.append(dict(initial_radius=initial_r, endpoint_velocity_scaled=float(end[1]),
                           shooting_derivative_scaled=float(D), integration_nodes=len(solution.t)))
        if abs(end[1]) < 1e-12:
            break
        initial_r -= end[1]/D
    assert abs(end[1]) < 1e-10
    knots = np.linspace(0, 1, count+1)
    values = solution.sol(knots)
    slopes = np.column_stack([rhs(t, values[:, i]) for i, t in enumerate(knots)])
    np.savez(REFERENCE, knots=knots, state=values[:3], slope=slopes[:3],
             fundamental=values[3:].T.reshape(-1, 3, 3), fundamental_slope=slopes[3:].T.reshape(-1, 3, 3),
             scale=scale, background_sha256=check_inputs()['background.npz'])
    return dict(rhs_evaluations=evaluations, reference_nodes=len(knots), trials=trials)


def zeros():
    return np.full((3, 3), iv.mpf(0), dtype=object)


def algebra_checks():
    x, s, delta = sp.symbols('x s delta')
    P = sp.Matrix([[1, x, 0], [x, 1+x*x, 0], [0, 0, 1+x]])
    e1 = sp.Matrix([1, 0, 0])
    terminal = P.subs(x, 1)
    ell = terminal[1, :]/terminal[1, 0]
    A0 = P.diff(x)*P.inv()
    checked = 0
    for component in range(3):
        for power in range(3):
            v = sp.zeros(3, 1)
            v[component] = x**power
            primitive = v.applyfunc(lambda a: sp.integrate(a.subs(x, s), (s, 0, x)))
            solution = P*(primitive-e1*(ell*primitive.subs(x, 1))[0]-e1*delta/terminal[1, 0])
            assert all(sp.simplify(a) == 0 for a in solution.diff(x)-A0*solution-P*v)
            assert solution[1].subs(x, 0) == solution[2].subs(x, 0) == 0
            assert sp.simplify(solution[1].subs(x, 1)+delta) == 0
            checked += 1
    r, z, q = sp.symbols('r z q')
    K, L = sp.Function('K')(r), sp.Function('L')(r)
    flow = sp.Matrix([z, (sp.diff(K, r)*q-K*(1+q)*L)/8, -(1+q)*L*z])
    first = 4*z*z-K*q
    assert sp.expand(sum(sp.diff(first, a)*v for a, v in zip((r, z, q), flow))) == 0
    return dict(exact_boundary_inverse_models=checked, exact_first_integral=True)


def matrix_abs(A):
    return np.array([[abs(v) for v in row] for row in A], dtype=object)


def inverse_box(A):
    midpoint = np.array([[float(v.mid) for v in row] for row in A])
    B = np.array([[exact(float(v)) for v in row] for row in np.linalg.inv(midpoint)], dtype=object)
    eta = interval_norm(np.eye(3, dtype=int)-B@A)
    assert eta < .4
    M = B@A
    for k in range(3):
        pivot = M[k, k]
        assert pivot.a > 0
        for j in range(3):
            B[k, j] /= pivot
            if j != k:
                M[k, j] /= pivot
        M[k, k] = iv.mpf(1)
        for i in range(3):
            if i == k:
                continue
            factor = M[i, k]
            for j in range(3):
                B[i, j] -= factor*B[k, j]
                if j != k:
                    M[i, j] -= factor*M[k, j]
            M[i, k] = iv.mpf(0)
    return B, eta


class OrbitCertificate:
    def __init__(self, bg, radius):
        self.bg, self.hashes = bg, check_inputs()
        self.background_knots = [Fraction(float(v)) for v in bg.sol.x]
        self.radius = iv.mpf(radius)
        self.scale = list(map(iv.mpf, SCALE_EXACT))
        self.tau = 5*iv.pi
        self.reference_hash = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
        with np.load(REFERENCE) as data:
            self.knots = [Fraction(float(v)) for v in data['knots']]
            self.state, self.slope, self.P, self.Pp = [data[k].copy() for k in ('state', 'slope', 'fundamental', 'fundamental_slope')]
            assert str(data['background_sha256']) == self.hashes['background.npz']
            assert np.array_equal(data['scale'], SCALE)
        assert len(self.knots) <= 4000 and self.knots[0] == 0 and self.knots[-1] == 1
        assert all(a < b for a, b in zip(self.knots[:-1], self.knots[1:]))
        assert self.state.shape == self.slope.shape == (3, len(self.knots))
        assert self.P.shape == self.Pp.shape == (len(self.knots), 3, 3)
        assert np.array_equal(self.P[0], np.eye(3)) and self.state[1, 0] == self.state[2, 0] == 0
        terminal = np.array([[exact(float(v)) for v in row] for row in self.P[-1]], dtype=object)
        self.D = terminal[1, 0]
        assert self.D.b < 0
        self.l = terminal[1]/self.D
        self.projection = np.eye(3, dtype=int)-np.outer([1, 0, 0], self.l)
        self.background_evaluations = 0

    def background_box(self, radius):
        lo, hi = Fraction(lower(radius)), Fraction(upper(radius))
        assert Fraction('15.65') < lo <= hi < Fraction('16.34')
        points = sorted({lo, hi} | {v for v in self.background_knots if lo < v < hi})
        boxes = []
        for a, b in zip(points[:-1], points[1:]):
            piece = bisect_right(self.background_knots, (a+b)/2)-1
            boxes.append(physical_cell(self.bg, piece, a, b))
            self.background_evaluations += 1
            if self.background_evaluations > 20000:
                raise RuntimeError('Azimuth verification exceeded 20000 background piece evaluations.')
        return [enclosure(min(v[j].a for v in boxes), max(v[j].b for v in boxes)) for j in range(4)]

    def radial_background(self, radius, order):
        r = Jet(0, [radius, iv.mpf(1)]+[iv.mpf(0)]*(order-1))
        y = [Jet(v, order=order) for v in self.background_box(radius)]
        for k in range(order):
            flow = field_equations(r, y, iv.mpf('.06'), iv.mpf('.08'), iv.pi)
            for j in range(4):
                y[j].c[k+1] = flow[j].c[k]/(k+1)
        return r, y

    def composed_background(self, r):
        _, radial = self.radial_background(r.c[0], r.order)
        delta = Jet(0, [iv.mpf(0)]+r.c[1:])
        values = []
        for v in radial:
            value = Jet(v.c[-1], order=r.order)
            for coefficient in reversed(v.c[:-1]):
                value = value*delta+coefficient
            values.append(value)
        return values

    def trial(self, cell, x):
        a, b = self.knots[cell:cell+2]
        y, yp = [], []
        for j in range(3):
            v, vp = hermite(x, a, b, self.state[j, cell:cell+2], self.slope[j, cell:cell+2])
            y.append(v)
            yp.append(vp)
        P, Pp = [], []
        for i in range(3):
            row, prime = [], []
            for j in range(3):
                v, vp = hermite(x, a, b, self.P[cell:cell+2, i, j], self.Pp[cell:cell+2, i, j])
                row.append(v)
                prime.append(vp)
            P.append(row)
            Pp.append(prime)
        return y, yp, P, Pp

    def residual(self, cell, point):
        a, b = self.knots[cell:cell+2]
        value = exact((a+b)/2) if point else enclosure(a, b)
        x = Jet(0, [value, iv.mpf(1), iv.mpf(0), iv.mpf(0)])
        y, yp, P, Pp = self.trial(cell, x)
        r, z, q = [v*s for v, s in zip(y, self.scale)]
        bg = self.composed_background(r)
        flow, jac, _, _ = physical_flow(r, z, q, bg)
        A = [[Jet.coerce(jac[i][j], 3)*self.tau*self.scale[j]/self.scale[i] for j in range(3)] for i in range(3)]
        defect = [yp[i]-flow[i]*self.tau/self.scale[i] for i in range(3)]
        matrix = [[Pp[i][j]-sum((A[i][k]*P[k][j] for k in range(3)), Jet(0, order=3)) for j in range(3)] for i in range(3)]
        return defect, matrix, P, y

    def nonlinear_jacobian(self, y):
        rho = enclosure(-self.radius, self.radius)
        r, bg = self.radial_background(y[0].c[0]+rho, 1)
        z = Jet((y[1].c[0]+rho)*self.scale[1], order=1)
        q = Jet((y[2].c[0]+rho)*self.scale[2], order=1)
        _, jac, dz, dq = physical_flow(r, z, q, bg)
        bound = zeros()
        for i in range(3):
            for j in range(3):
                Jr = Jet.coerce(jac[i][j], 1).c[1]
                Jz = Jet.coerce(dz[i][j], 1).c[0]
                Jq = Jet.coerce(dq[i][j], 1).c[0]
                bound[i, j] = self.tau*self.scale[j]/self.scale[i]*self.radius*(abs(Jr)+self.scale[1]*abs(Jz)+self.scale[2]*abs(Jq))
        return bound

    def cell(self, index):
        a, b = self.knots[index:index+2]
        center = self.residual(index, True)
        ranged = self.residual(index, False)
        offset = enclosure(-(b-a)/2, (b-a)/2)
        def bound(one, all_values):
            value = sum((one.c[k]*offset**k for k in range(3)), iv.mpf(0))+all_values.c[3]*offset**3
            lo, hi = max(value.a, all_values.c[0].a), min(value.b, all_values.c[0].b)
            assert lo <= hi
            return abs(enclosure(lo, hi))
        residual = np.array([bound(u, v) for u, v in zip(center[0], ranged[0])], dtype=object)
        defect = np.array([[bound(center[1][i][j], ranged[1][i][j]) for j in range(3)] for i in range(3)], dtype=object)
        P = np.array([[v.c[0] for v in row] for row in ranged[2]], dtype=object)
        inverse, eta = inverse_box(P)
        difference = defect@matrix_abs(inverse)+self.nonlinear_jacobian(ranged[3])
        return dict(P=P, inverse=inverse, residual=residual, difference=difference,
                    inverse_precondition_defect=eta, interval=[a, b])


def implementation_checks(obj):
    maximum, checks = 0., 0
    for radius in (15.69, 15.9, 16.1, 16.28):
        r, bg = obj.radial_background(exact(radius), 1)
        # Point data isolate the formula test from background interval width.
        for j, value in enumerate(obj.bg.sol(radius)):
            bg[j].c[0] = exact(float(value))
        rhs = field_equations(r, bg, iv.mpf('.06'), iv.mpf('.08'), iv.pi)
        for j in range(4):
            bg[j].c[1] = rhs[j].c[0]
        z, q = Jet('.04', order=1), Jet('.00007', order=1)
        flow, jac, dz, dq = physical_flow(r, z, q, bg)
        for i in range(3):
            assert abs(Jet.coerce(flow[i], 1).c[1]-Jet.coerce(jac[i][0], 1).c[0]).b < iv.mpf('1e-23')
            checks += 1
        for component, derivative in ((1, dz), (2, dq)):
            rr, yy = Jet(radius, order=1), [Jet(float(v), order=1) for v in obj.bg.sol(radius)]
            zz, qq = Jet('.04', order=1), Jet('.00007', order=1)
            (zz if component == 1 else qq).c[1] = iv.mpf(1)
            shifted, matrix, _, _ = physical_flow(rr, zz, qq, yy)
            for i in range(3):
                assert abs(Jet.coerce(shifted[i], 1).c[1]-Jet.coerce(jac[i][component], 1).c[0]).b < iv.mpf('1e-23')
                checks += 1
                for j in range(3):
                    assert abs(Jet.coerce(matrix[i][j], 1).c[1]-Jet.coerce(derivative[i][j], 1).c[0]).b < iv.mpf('1e-23')
                    checks += 1
        g = obj.bg.fields(np.array([radius]))
        K = radius*radius*g['f'][0]
        Kp = 2*radius*g['f'][0]+radius*radius*g['fp'][0]
        L = 2*g['nup'][0]-2/radius
        expected = [.04, (Kp*.00007-K*1.00007*L)/8, -1.00007*L*.04]
        maximum = max(maximum, *(abs(float(v.c[0].mid)-w) for v, w in zip(flow, expected)))
    assert maximum < 2e-14
    inverse_checks = 0
    nominal = sp.Matrix([[2, 1, 0], [-1, 3, 1], [1, 0, 2]])
    width = sp.Rational(1, 10000)
    box = np.array([[enclosure(Fraction(v-width), Fraction(v+width)) for v in row]
                    for row in nominal.tolist()], dtype=object)
    bounded, _ = inverse_box(box)
    for pattern in range(8):
        matrix = nominal+sp.Matrix(3, 3, lambda i, j:
                                  width*(-1 if pattern & (1 << ((i+2*j)%3)) else 1))
        inverse = matrix.inv()
        for i in range(3):
            for j in range(3):
                value = exact(Fraction(inverse[i, j]))
                assert bounded[i, j].a <= value.a and value.b <= bounded[i, j].b
                inverse_checks += 1
    return dict(ode_derivative_checks=checks, independent_flow_checks=12,
                maximum_independent_flow_difference=maximum, rational_inverse_components=inverse_checks)


def identify_branch(obj, error):
    relative = RelativePotential.load(obj.bg)
    eps = list(map(exact, error))
    radius_bounds, initial_acceleration, final_acceleration = [], None, None
    min_interior_velocity, min_weight, min_f = float('inf'), float('inf'), float('inf')
    for i in range(len(obj.knots)-1):
        a, b = obj.knots[i:i+2]
        trial, *_ = obj.trial(i, Jet(enclosure(a, b), order=1))
        reference = [v.c[0]*s for v, s in zip(trial, obj.scale)]
        r, z, q = [v+enclosure(-e, e) for v, e in zip(reference, eps)]
        radius_bounds.append((lower(r), upper(r)))
        min_weight = min(min_weight, lower(1+reference[2]))
        rr, bg = obj.radial_background(r, 1)
        min_f = min(min_f, lower(1-2*bg[2].c[0]))
        if i == 0 or i == len(obj.knots)-2:
            flow = physical_flow(rr, Jet(z, order=1), Jet(q, order=1), bg)[0]
            if i == 0:
                initial_acceleration = lower(flow[1].c[0])
                assert initial_acceleration > 0
            else:
                final_acceleration = upper(flow[1].c[0])
                assert final_acceleration < 0
        else:
            min_interior_velocity = min(min_interior_velocity, lower(z))
    assert min_interior_velocity > 0 and min_weight > 0 and min_f > 0
    r0 = exact(float(obj.state[0, 0]))+enclosure(-eps[0], eps[0])
    left, right = Fraction(lower(r0)), Fraction(upper(r0))
    points = sorted({left, right} | {v for v in relative.nodes if left < v < right})
    boxes = [relative.fraction(a, b)[0] for a, b in zip(points[:-1], points[1:])]
    fraction = enclosure(min(v.a for v in boxes), max(v.b for v in boxes))
    assert fraction.a > iv.mpf('.60') and fraction.b < iv.mpf('.62')
    assert r0.b < exact(relative.ring_brackets[0][0]).a
    extent = [min(v[0] for v in radius_bounds), max(v[1] for v in radius_bounds)]
    assert exact(extent[0]) > iv.mpf('15.67') and exact(extent[1]) < iv.mpf('16.31')
    return dict(status='single_outward_resonant_half_orbit', radial_extent=extent,
                energy_fraction=[lower(fraction), upper(fraction)],
                first_cell_acceleration_lower=initial_acceleration,
                last_cell_acceleration_upper=final_acceleration,
                interior_velocity_lower=min_interior_velocity, reference_weight_lower=min_weight,
                metric_f_lower=min_f, weight_error_upper=error[2],
                normalized_measure_error_upper=upper(2*eps[2]),
                relative_optics_sha256=relative.record_sha256)


def run_certificate(obj, max_seconds):
    start = perf_counter()
    rows = []
    for i in range(len(obj.knots)-1):
        if perf_counter()-start > max_seconds:
            raise RuntimeError('Azimuth verification stopped at its time budget.')
        rows.append(obj.cell(i))
    C, V, R = [zeros()], [zeros()], [np.full(3, iv.mpf(0), dtype=object)]
    for row in rows:
        width = exact(row['interval'][1]-row['interval'][0])
        inv = row['inverse']
        C.append(C[-1]+matrix_abs(inv)*width)
        V.append(V[-1]+matrix_abs(obj.projection@inv)*width)
        R.append(R[-1]+matrix_abs(np.array([obj.l@inv], dtype=object))[0]*width)
    G, residual, difference, lift = zeros(), np.full(3, iv.mpf(0), dtype=object), zeros(), np.full(3, iv.mpf(0), dtype=object)
    for i, row in enumerate(rows):
        P = row['P']
        first = matrix_abs(P@obj.projection)@C[i+1]
        alternate = matrix_abs(P)@V[i+1]
        tail = np.array([[abs(P[j, 0])*v for v in R[-1]-R[i]] for j in range(3)], dtype=object)
        for j in range(3):
            residual[j] = exact(max(upper(residual[j]), upper(row['residual'][j])))
            lift[j] = exact(max(upper(lift[j]), upper(abs(P[j, 0]*exact(float(obj.state[1, -1]))/obj.D))))
            for k in range(3):
                gain = exact(min(upper(first[j, k]), upper(alternate[j, k])))+tail[j, k]
                G[j, k] = exact(max(upper(G[j, k]), upper(gain)))
                difference[j, k] = exact(max(upper(difference[j, k]), upper(row['difference'][j, k])))
    N = G@difference
    eta = exact(interval_norm(N))
    drive = G@residual+lift
    beta = exact(max(upper(v) for v in drive))
    image = beta+eta*obj.radius
    closed = eta.b < 1 and image.b < obj.radius.a
    result = dict(status='azimuth_orbit_ball_verified' if closed else 'azimuth_orbit_ball_not_closed',
                  radius=upper(obj.radius), contraction_upper=upper(eta), drive_upper=upper(beta), ball_image_upper=upper(image),
                  scale=list(SCALE_EXACT), cells=len(rows), background_piece_evaluations=obj.background_evaluations,
                  green_component_bounds=[[upper(v) for v in row] for row in G],
                  residual_component_bounds=[upper(v) for v in residual],
                  linearization_difference_bounds=[[upper(v) for v in row] for row in difference],
                  boundary_lift_bounds=[upper(v) for v in lift],
                  reference_boundary_denominator=[lower(obj.D), upper(obj.D)],
                  maximum_inverse_precondition_defect=max(row['inverse_precondition_defect'] for row in rows))
    if closed:
        summed, term = drive.copy(), drive.copy()
        for _ in range(24):
            term = N@term
            summed += term
        remainder = eta**25*beta/(1-eta)
        error = [upper((v+remainder)*s) for v, s in zip(summed, obj.scale)]
        result['physical_error_bounds'] = error
        result['scaled_error_bounds'] = [upper(v+remainder) for v in summed]
    result['elapsed_seconds'] = perf_counter()-start
    return result


def observation_budget(report):
    path = ROOT/'data/iterated-tidal-observation-50.json'
    previous = json.loads(path.read_text())
    assert previous['status'] == 'iterated_tidal_field_observation_enclosed'
    assert previous['input_sha256'] == report['input_sha256']
    assert previous['relative_optics_sha256'] == report['branch']['relative_optics_sha256']
    for name, digest in previous['dependency_sha256'].items():
        assert hashlib.sha256((ROOT.parent.parent/name).read_bytes()).hexdigest() == digest
    assert previous['interval']['xi'] == ['3/5', '31/50']
    assert previous['interval']['radial_interval'] == ['15.67', '16.31']
    amplitude = exact(previous['interval']['trial_amplitude_bounds'][2])
    field = exact(previous['interval']['field_observation_error_bounds'][2])
    weight = exact(report['branch']['weight_error_upper'])
    return dict(iterated_observation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                weight_contribution_upper=upper(2*amplitude*weight),
                field_and_weight_contribution_upper=upper(field+2*amplitude*weight),
                remaining_terms=['trial_amplitude_geometry_error', 'reference_quadrature_error'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--cells', type=int, choices=(128, 256, 512), default=256)
    parser.add_argument('--digits', type=int, choices=(30, 50), default=30)
    parser.add_argument('--radius', default='.0005')
    parser.add_argument('--output', type=Path, default=ROOT/'data/azimuth-orbit.json')
    args = parser.parse_args()
    iv.dps = args.digits
    bg = Background()
    proposal = build_reference(bg, args.cells) if args.build or args.build_only or not REFERENCE.exists() else None
    if args.build_only:
        print(json.dumps(proposal, indent=2))
        return
    obj = OrbitCertificate(bg, args.radius)
    checks = dict(algebra=algebra_checks(), implementation=implementation_checks(obj))
    report = run_certificate(obj, 240)
    if report['status'] == 'azimuth_orbit_ball_verified':
        report['branch'] = identify_branch(obj, report['physical_error_bounds'])
    report.update(digits=args.digits, input_sha256=obj.hashes, reference_sha256=obj.reference_hash,
                  checks=checks, total_background_piece_evaluations=obj.background_evaluations,
                  scope='Boundary correction for the azimuth orbit; tidal observation and radiation are not certified.')
    if 'branch' in report:
        report['observation_budget'] = observation_budget(report)
    dependencies = ['check_barrier_certificate.py', 'check_background_defect.py',
                    'check_polar_coercivity.py', 'check_polar_transverse.py',
                    'check_finite_static_inverse.py', 'check_asymptotic_exterior.py',
                    'check_relative_optics.py']
    report['dependency_sha256'] = {str((OLD/name).relative_to(ROOT.parent.parent)):
                                   hashlib.sha256((OLD/name).read_bytes()).hexdigest() for name in dependencies}
    if proposal:
        report['proposal'] = proposal
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='ascii')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
