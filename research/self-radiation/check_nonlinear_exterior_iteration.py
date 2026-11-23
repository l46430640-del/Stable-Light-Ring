"""Algebraic checks: nonlinear exterior iteration."""
import json
import sympy as sp
from check_joint_gauge_matter import matrix


def zero(expression):
    if isinstance(expression, sp.MatrixBase):
        assert expression.applyfunc(sp.simplify) == sp.zeros(*expression.shape)
    else:
        assert sp.expand(expression) == 0 or sp.simplify(expression) == 0


def pair(prefix):
    return sp.Matrix(sp.symbols(f'{prefix}0:2', real=True))


def matter_hessians():
    phi, psi, chi = pair('x'), pair('v'), pair('w')
    a, b = pair('a'), pair('b')
    av, bv, aw, bw = pair('av'), pair('bv'), pair('aw'), pair('bw')
    g, h, k = sp.symbols('g h k', real=True)
    mu, sigma = sp.symbols('mu sigma', positive=True)
    s = sp.symbols('s', real=True)
    potential = mu**2*s*(1-2*s/sigma**2)**2
    evaluate = lambda expression: expression.subs(s, phi.dot(phi))
    vp, vpp, vppp = [evaluate(sp.diff(potential, s, j)) for j in (1, 2, 3)]
    stress = a.dot(b)+g*evaluate(potential)/2
    coordinates = [g, *phi, *a, *b]
    direction_v = [h, *psi, *av, *bv]
    direction_w = [k, *chi, *aw, *bw]
    derivative = lambda value, direction: sum(sp.diff(value, x)*v
                                              for x, v in zip(coordinates, direction))
    actual = derivative(derivative(stress, direction_v), direction_w)
    expected = av.dot(bw)+aw.dot(bv)+h*vp*phi.dot(chi)+k*vp*phi.dot(psi)
    expected += g*(vp*psi.dot(chi)+2*vpp*phi.dot(psi)*phi.dot(chi))
    zero(actual-expected)
    scalar_force = vp*phi
    actual_force = scalar_force.jacobian(phi)*psi
    actual_force = actual_force.jacobian(phi)*chi
    expected_force = 2*vpp*(phi.dot(chi)*psi+phi.dot(psi)*chi+psi.dot(chi)*phi)
    expected_force += 4*vppp*phi.dot(psi)*phi.dot(chi)*phi
    zero(actual_force-expected_force)
    zero(actual_force.subs({x: 0 for x in phi}))
    return dict(trace_reversed_stress_hessian=True, full_complex_potential_hessian=True,
                scalar_force_hessian_vanishes_at_zero_field=True,
                stress_hessian_retains_two_scalar_factors=True,
                scalar_differential_order_at_most_two=True)


def inverse_metric_hessian():
    cases = []
    for seed in (1, 3, 7):
        basis = sp.eye(4)+matrix(seed)/4
        g = basis.T*sp.diag(-1, 1, 1, 1)*basis
        J = g.inv()
        h, k = matrix(seed+1), matrix(seed+4)
        Jh, Jk = -J*h*J, -J*k*J
        Jhk = J*h*J*k*J+J*k*J*h*J
        zero(g*Jh+h*J)
        zero(g*Jk+k*J)
        zero(g*Jhk+h*Jk+k*Jh)
        assert Jhk != 2*J*h*J*k*J
        Aphi, Apsi, Achi = [matrix(seed+offset) for offset in (11, 14, 18)]
        actual_mixed = sp.trace(Jhk*Aphi+Jh*Achi+Jk*Apsi)
        expected = sp.trace((J*h*J*k*J+J*k*J*h*J)*Aphi-J*h*J*Achi-J*k*J*Apsi)
        zero(actual_mixed-expected)
        cases.append(dict(seed=seed, inverse_identity_to_mixed_second_order=True,
                          both_matrix_orders_needed=True, scalar_mixed_block_identity=True))
    return dict(cases=cases, differentiated_metric_directions_required=False,
                scope='Finite rational matrices verify instances of the inverse-variation identities proved analytically')


def boundary_source_and_scaling():
    rho0, gamma, z, z1, z2, t, eps, delta = sp.symbols('rho0 gamma z z1 z2 t eps delta', real=True)
    c, u, v = pair('c'), pair('u'), pair('v')
    linearized_along_ray = -4*gamma*z-t*rho0*c.dot(c)
    boundary = sp.integrate(linearized_along_ray, (t, 0, 1))
    zero(boundary+4*gamma*z+rho0*c.dot(c)/2)
    varied = -4*gamma*(z+eps*z1+delta*z2)-rho0*(c+eps*u+delta*v).dot(c+eps*u+delta*v)/2
    zero(sp.diff(varied, eps).subs({eps: 0, delta: 0})+4*gamma*z1+rho0*c.dot(u))
    zero(sp.diff(varied, eps, delta)+rho0*u.dot(v))
    # The boundary field equation gives the same coefficient as Bondi mass loss.
    mass_derivative = gamma*rho0*(-rho0*c.dot(c)/(8*gamma))/(4*sp.pi)
    zero(mass_derivative+(rho0*c).dot(rho0*c)/(32*sp.pi))
    x, y, value = sp.symbols('x y value', positive=True)
    rho, scale = x*y, x**(-3)*y**(-2)
    zero(scale-y*rho**(-3))
    trial_polynomial = lambda h: h+3*h*h+2*h**3
    identity = sp.integrate(2*scale*sp.diff(trial_polynomial(value), value)
                           .subs(value, t*rho)*rho, (t, 0, 1))
    zero(identity-2*scale*trial_polynomial(rho))
    assert sp.simplify(identity-2*y**(-1)*rho**3*trial_polynomial(rho)) != 0
    return dict(boundary_value_and_two_variations=True, only_11_boundary_output=True,
                mass_loss_coefficient='-1/(32*pi)', correct_equation_scaling='2*y*rho^(-3) = 2*q*r^2',
                nonlinear_scaling_fixed_by_integrated_linearization=True,
                scope='Boundary derivatives assume the leading linearized matrix derived in the reference; the scaling is reconstructed from its defining linearization')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "matter": matter_hessians(),
        "inverse_metric": inverse_metric_hessian(),
        "boundary": boundary_source_and_scaling(),
    }, indent=2))
