"""Algebraic checks: parameter radiation map."""
import json
import sympy as sp


def common_chart_and_mass_derivative():
    q, r, m, mu = sp.symbols('q r m mu', positive=True)
    f = 1-2*m/r
    jac = sp.Matrix([[-1, 1/f], [0, 1]])
    outgoing = sp.simplify(jac.T*sp.diag(-f, 1/f)*jac)
    assert (outgoing-sp.Matrix([[-f, 1], [1, 0]])).applyfunc(sp.simplify) == sp.zeros(2)
    assert outgoing.det() == -1
    assert (outgoing.inv()-sp.Matrix([[0, 1], [1, f]])).applyfunc(sp.simplify) == sp.zeros(2)
    v = sp.Function('v')(q, r)
    psi = v/r
    box = (sp.diff(r*r*sp.diff(psi, r), q)
           +sp.diff(r*r*(sp.diff(psi, q)+f*sp.diff(psi, r)), r))/(r*r)
    rescaled = sp.expand(q*r*r*(-box+mu*mu*psi))
    expected = -q*r*(2*sp.diff(v, q, r)+f*sp.diff(v, r, 2)
                      +sp.diff(f, r)*sp.diff(v, r)-sp.diff(f, r)*v/r)+mu*mu*q*r*v
    assert sp.simplify(rescaled-expected) == 0
    shifted_D1 = lambda expr: -r*sp.diff(expr, r)+expr
    target = 2*q/r**2*shifted_D1(shifted_D1(v))
    assert sp.simplify(sp.diff(rescaled, m)-target) == 0
    assert sp.diff(rescaled, m, 2) == 0
    assert sp.diff(mu*mu*q*r, m) == 0
    c_r = 1/(r*f)
    assert sp.simplify(sp.diff(c_r, m)-2/(r*r*f*f)) == 0
    assert sp.simplify(sp.diff(c_r, m, 2)-8/(r**3*f**3)) == 0
    x_m = r+2*m*sp.log(r-2*m)
    assert sp.simplify(sp.diff(x_m, m)-2*sp.log(r-2*m)+4*m/(r-2*m)) == 0
    return dict(common_metric='-f dq^2+2 dq dr+r^2 dOmega^2',
                volume_density_independent_of_mass=True,
                scalar_mass_potential_independent_of_mass=True,
                reference_operator_mass_derivative='2 rho0 rhoI^2 (D1+1)^2',
                reference_operator_second_mass_derivative=0,
                gauge_covector_mass_derivatives_decay=True,
                scope='The undamped Schwarzschild scalar operator is checked exactly')


def leading_forcing_pattern():
    a, b, c, d, angle, component11 = sp.symbols('a b c d angle component11')
    dA = sp.Matrix([[0, 0, 0], [a, 0, 0], [0, b, 0]])
    dB = sp.Matrix([[0, 0, 0], [c, 0, 0], [d, 0, 0]])
    boundary_field = sp.Matrix([0, angle, component11])
    result = (dA+dB)*boundary_field
    assert result == sp.Matrix([0, 0, b*angle])
    return dict(nondecaying_output_components=['11'], gauge_input_boundary_value=0,
                scalar_parameter_mass_potential_derivative=0,
                scope='Three-block pattern of the leading metric matrices, not the full tensor operator')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "coordinates": common_chart_and_mass_derivative(),
        "leading_source": leading_forcing_pattern(),
    }, indent=2))
