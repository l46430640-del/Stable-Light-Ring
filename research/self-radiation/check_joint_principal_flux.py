"""Algebraic checks: joint principal flux."""
import json
import sympy as sp


def zero(expression):
    assert sp.expand(expression) == 0 or sp.simplify(expression) == 0


def symmetric_matrix(prefix):
    entries = {(i, j): sp.Symbol(f'{prefix}{i}{j}', real=True)
               for i in range(4) for j in range(i, 4)}
    return sp.Matrix(4, 4, lambda i, j: entries[min(i, j), max(i, j)])


def principal_cancellation():
    inverse = symmetric_matrix('G')
    h = symmetric_matrix('h')
    xi = sp.Matrix(sp.symbols('xi0:4', real=True))
    gamma = [[[sum(inverse[k, a]*(xi[i]*h[j, a]+xi[j]*h[i, a]
                                 -xi[a]*h[i, j])/2 for a in range(4))
               for j in range(4)] for i in range(4)] for k in range(4)]
    trace = sum(inverse[a, b]*h[a, b] for a in range(4) for b in range(4))
    constraint = [sum(inverse[a, b]*xi[a]*h[b, i]
                      for a in range(4) for b in range(4))-xi[i]*trace/2
                  for i in range(4)]
    wave = (xi.T*inverse*xi)[0]
    for i in range(4):
        for j in range(4):
            ricci = sum(xi[k]*gamma[k][i][j]-xi[j]*gamma[k][i][k]
                        for k in range(4))
            reduced = ricci-(xi[i]*constraint[j]+xi[j]*constraint[i])/2
            zero(reduced+wave*h[i, j]/2)
    a, rho = sp.symbols('a rho', positive=True)
    zero(2*a*(-wave/2)*rho-(-a*wave*rho))
    return dict(metric_components_checked=16, inverse_metric='arbitrary symmetric matrix',
                perturbation='arbitrary symmetric matrix', gauge_constraint_used=False,
                rescaled_principal_operator='-q*r*g^{ab} partial_a partial_b I_12',
                matter_orders={'scalar_to_metric': 1, 'metric_to_scalar': 0},
                scope='Principal Ricci-gauge cancellation; matter orders follow the separately checked full reduction')


def principal_frame():
    q, r, f, y, rho0 = sp.symbols('q r f y rho0', positive=True)
    # Columns are the edge frame in the (q,r,angular) coordinate frame.
    edge = sp.Matrix([[-q, 0, 0, 0], [-r, -r, 0, 0],
                      [0, 0, sp.sqrt(q/r), 0], [0, 0, 0, sp.sqrt(q/r)]])
    coframe = sp.Matrix([[-1, 2/f, 0, 0], [-1, 0, 0, 0],
                         [0, 0, r, 0], [0, 0, 0, r]])
    B = sp.Matrix([[y-2/f, -2/f, 0, 0], [y, 0, 0, 0],
                   [0, 0, sp.sqrt(y), 0], [0, 0, 0, sp.sqrt(y)]])
    for value in coframe*edge/r-B.subs(y, q/r):
        zero(value)
    null_metric = sp.Matrix([[0, -f/2, 0, 0], [-f/2, 0, 0, 0],
                             [0, 0, 1, 0], [0, 0, 0, 1]])
    reference = sp.Matrix([[2-f*y, 1, 0, 0], [1, 0, 0, 0],
                           [0, 0, 1, 0], [0, 0, 0, 1]])
    for value in B.T*null_metric*B/y-reference:
        zero(value)
    H = symmetric_matrix('H')
    correction = rho0*B.T*H*B
    h11_part = correction.diff(H[1, 1])
    expected = sp.zeros(4)
    expected[0, 0] = rho0*y*y
    for value in h11_part-expected:
        zero(value)
    for i in (2, 3):
        for j in (2, 3):
            zero(correction[i, j]-rho0*y*H[i, j])
    # The boundary matrix permits only H11 and angular trace-free entries.
    boundary = sp.zeros(4)
    boundary[1, 1] = H[1, 1]
    boundary[2, 2], boundary[3, 3] = H[2, 2], -H[2, 2]
    boundary[2, 3] = boundary[3, 2] = H[2, 3]
    for value in B.T*boundary*B:
        assert sp.limit(value, y, 0, dir='+') == 0
    V = sp.Matrix([1, -2, 0, 0])
    dtau = sp.Matrix([0, -1, 0, 0])
    zero((V.T*reference*V)[0]+2+f*y)
    zero((dtau.T*reference.inv()*dtau)[0]+2-f*y)
    zero((dtau.T*V)[0]-2)
    assert reference.det() == -1
    return dict(coframe_matrix_identity=True, full_reference_frame_identity=True,
                H11_contribution='rho0*y^2*H11 in entry 00 only',
                angular_contribution='rho0*y*H_AB',
                perturbative_principal_weight='rho0^(1+ell0)*y^ellI, 0<ellI<1/2',
                timelike_vector_norm='-(2+f*y)', timelike_covector_norm='-(2-f*y)',
                V_tau=2, reference_frame_determinant=-1,
                note='Pointwise weights and tame inverse-matrix bounds also require the stated analytical regularity assumptions')


def positive_flux_and_mass_duality():
    v0, b, M, z, p0, p1, p2, p3 = sp.symbols('v0 b M z p0 p1 p2 p3', real=True)
    twice_flux = v0*(p0*p0+p1*p1+p2*p2+p3*p3+M*z*z)+2*b*p0*p1
    squares = (v0-b)*(p0*p0+p1*p1)+b*(p0+p1)**2+v0*(p2*p2+p3*p3+M*z*z)
    zero(twice_flux-squares)
    zeta = sp.symbols('zeta', positive=True)
    ur, ui, zr, zi = sp.symbols('ur ui zr zi', real=True)
    for sign in (1, -1):
        gap = zeta*M*(ur*ur+ui*ui)+M*(zr*zr+zi*zi)/zeta-sign*2*M*(ur*zr+ui*zi)
        zero(gap-M*((zr-sign*zeta*ur)**2+(zi-sign*zeta*ui)**2)/zeta)
    x, y, mu = sp.symbols('rho0 y mu', positive=True)
    alpha0, alphaI = sp.symbols('alpha0 alphaI', real=True)
    V = lambda value: x*sp.diff(value, x)-2*y*sp.diff(value, y)
    J = sp.Function('J')(x, y)
    mass = mu**2/(x*x*y)
    weight_squared = x**(-2*alpha0)*y**(-2*alphaI)
    zero(V(mass))
    zero(sp.simplify(-V(weight_squared*mass*J)/(weight_squared*mass*J)
                     -(2*alpha0-4*alphaI-V(J)/J)))
    return dict(principal_flux_sum_of_squares=True,
                positivity_assumptions='V0>=b>=0, M>=0, future normal',
                complex_mass_boundary_young_identity=True, positive_and_negative_cross_terms=True,
                mass_bulk_coefficient='2*alpha0-4*alphaI-V(log J)',
                future_measure='w^2 i_V(J dmu_b), with outward orientation',
                boundary_derivative_requirement='z only', volume_derivative_requirement='z and Vz',
                no_bound_on_Mz_in_Y_assumed=True)


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "principal": principal_cancellation(),
        "frame": principal_frame(),
        "mass_flux": positive_flux_and_mass_duality(),
    }, indent=2))
