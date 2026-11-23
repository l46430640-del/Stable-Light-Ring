"""Algebraic checks: radiation trace recovery."""
import json
from fractions import Fraction
import sympy as sp


def zero(expression):
    assert sp.simplify(expression) == 0


def weighted_sources():
    ell0, gap = sp.symbols('ell0 gap', positive=True)
    ellI = sp.Rational(1, 2)-gap
    A = sp.Rational(1, 4)
    weak = (ell0, -A)
    off = (ell0+5, 3-A)
    diagonal = (2*ell0+12, 7-2*A)
    metric_target = (ell0, ellI)
    scalar_target = (ell0+4, 2-A)
    margins = []
    for coefficient in (off, diagonal):
        source = tuple(a+b for a, b in zip(coefficient, weak))
        for target in (metric_target, scalar_target):
            difference = tuple(a-b for a, b in zip(source, target))
            assert all(value.is_positive for value in difference)
            margins.append([str(value) for value in difference])
    y, eta, exponent = sp.symbols('y eta exponent', positive=True)
    integral = sp.integrate(y**(2*exponent-1), (y, 0, eta))
    zero(integral-eta**(2*exponent)/(2*exponent))
    # A weak metric field need not belong to the auxiliary sharp-proof space.
    witness_power = -A/2
    original_integrability = 2*(A+witness_power)
    auxiliary_A = sp.Rational(1, 16)
    auxiliary_integrability = 2*(auxiliary_A+witness_power)
    assert original_integrability > 0 and auxiliary_integrability < 0
    witness_norm = (1+witness_power**2)/original_integrability
    assert witness_norm == sp.Rational(65, 16)
    return dict(all_source_weight_margins_positive=True, margins=margins,
                normal_extension_norm_squared='eta^(2*zeta)/(2*zeta)',
                source_boundary_term_needs_only_A_positive=True,
                weak_witness='chi(q)*y^(-1/8), A=1/4',
                weak_radial_norm_squared=str(witness_norm),
                auxiliary_weight_countercheck='The witness is not in y^(-1/16) L2_b',
                identification='Construct the sharp metric solution separately; identify in the original weak space by uniqueness',
                scope='The witness is a function-space example, not a solution of the Einstein equations')


def source_interpolation():
    cases = []
    order, split = sp.symbols('p j', positive=True)
    for mixed in (False, True):
        low_field, reserve, low_background = (4, 6, 12) if mixed else (3, 5, 10)
        zero(split/order-(split-reserve)/(order-reserve)
             -reserve*(order-split)/(order*(order-reserve)))
        zero((order-split)/order-(order-split-low_field)/(order-low_field)
             -low_field*split/(order*(order-low_field)))
        checked = 0
        for p in range(low_field, 65):
            for j in range(p+1):
                background_order = j+reserve
                field_order = p-j
                if p <= reserve:
                    assert background_order <= low_background
                    assert field_order <= p
                else:
                    theta_b = max(Fraction(0), Fraction(background_order-low_background,
                                                       p+reserve-low_background))
                    theta_w = max(Fraction(0), Fraction(field_order-low_field, p-low_field))
                    assert 0 <= theta_b <= 1 and 0 <= theta_w <= 1
                    assert theta_b+theta_w <= 1
                checked += 1
        cases.append(dict(mixed=mixed, leibniz_splits_checked=checked,
                          low_field_order=low_field, low_background_order=low_background,
                          high_background_order=f'p+{reserve}'))
    return dict(cases=cases, general_symbolic_exponent_bounds=True,
                conclusion='Each coefficient-field product is bounded by low-background*high-field + high-background*low-field',
                assumptions='Sobolev coefficient bounds and interpolation in the stated weighted spaces',
                new_high_norm_smallness_required=False)


def radial_trace_coordinates():
    q, r, x, y = sp.symbols('q r rho0 y', positive=True)
    substitution = {x: 1/q, y: q/r}
    for coordinate, coefficient in ((1/q, x*x), (q/r, -x*y)):
        zero(-sp.diff(coordinate, q)-coefficient.subs(substitution))
    z, radial = sp.symbols('z radial', real=True)
    logarithmic_map = sp.Matrix([-z, z-radial])
    jacobian = logarithmic_map.jacobian([z, radial])
    assert jacobian.det() == 1
    assert jacobian.inv() == sp.Matrix([[-1, 0], [-1, -1]])
    # Exact half-line H1 estimate for a decaying oscillatory scalar mode.
    examples = []
    for decay in (Fraction(1, 4), Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4)):
        for frequency in (0, 1, 3, 8):
            integrated_H1 = (1+decay*decay+frequency*frequency)/(2*decay)
            gap = ((decay-1)**2+frequency*frequency)/(2*decay)
            assert integrated_H1-1 == gap >= 0
            examples.append(dict(decay=str(decay), frequency=frequency,
                                 value_at_origin_squared='1', H1_norm_squared=str(integrated_H1)))
    parameter_orders = []
    for j in range(5):
        for k in range(5):
            if j+k < 2:
                continue
            boundary_order = j+k+2
            radius_order = j+k+3
            assert boundary_order-1 >= j+k+1
            assert radius_order-2 >= j+k+1
            assert radius_order <= 48
            parameter_orders.append(dict(parameter_derivatives=j, tangential_order=k,
                                         limiting_news_order=boundary_order,
                                         finite_radius_convergence_order=radius_order))
    assert parameter_orders[-1]['limiting_news_order'] == 10
    assert parameter_orders[-1]['finite_radius_convergence_order'] == 11
    return dict(fixed_radius_derivative='rho0*(D0-D1)', boundary_derivative='rho0*D0',
                logarithmic_coordinate_determinant=1, half_line_models=examples,
                parameter_orders=parameter_orders,
                rate='r^(-ellI) on fixed compact retarded-time windows',
                finite_radius_energy_flux_identified=False,
                scope='Coordinate and norm identities; the trace estimate is a Hilbert-valued Sobolev argument in the proof')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "weights": weighted_sources(),
        "interpolation": source_interpolation(),
        "traces": radial_trace_coordinates(),
    }, indent=2))
