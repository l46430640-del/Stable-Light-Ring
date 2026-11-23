"""Algebraic checks: individual quadrupole gain."""
from fractions import Fraction
import json
import sympy as sp


def zero(value):
    if isinstance(value, sp.MatrixBase):
        assert value.applyfunc(sp.simplify) == sp.zeros(*value.shape)
    else:
        assert sp.simplify(value) == 0


def projected_clock_jets():
    a, u, t1, t2, mass_shift = sp.symbols('a u t1 t2 mass_shift', real=True)
    projection = sp.diag(0, 1, 0)
    n1 = sp.Matrix([sp.Function('f')(u), 0, sp.Function('g')(u)])
    n2 = sp.Matrix([sp.Function(f'h{j}')(u) for j in range(3)])
    n3 = sp.Matrix([sp.Function(f'k{j}')(u) for j in range(3)])
    shift = t1*a+t2*a*a
    field = (a*(n1-shift*n1.diff(u)+shift**2*n1.diff(u, 2)/2)
             +a*a*(n2-shift*n2.diff(u))+a**3*n3)
    jets = [field.applyfunc(lambda value: sp.expand(value).coeff(a, j)) for j in range(1, 4)]
    zero(projection*jets[0])
    zero(projection*(jets[1]-n2))
    zero(projection*(jets[2]-n3)+t1*projection*n2.diff(u))
    zero(projection*(n3-mass_shift*n1.diff(u))-projection*n3)
    assert projection*n2.diff(u) != sp.zeros(3, 1)
    return dict(first_order_projection_zero=True, second_order_clock_invariant=True,
                cubic_projected_clock_term='-tau1*Pi*N2_prime',
                mass_log_correction_has_zero_quadrupole=True,
                scope='Constant clock shift and a fixed angular projection; no general supertranslation invariance is claimed')


def integral_remainder_and_constants():
    a, t = sp.symbols('a t', real=True)
    checked = 0
    for power in range(3, 10):
        third = power*(power-1)*(power-2)*(t*a)**(power-3)
        remainder = a**3*sp.integrate((1-t)**2*third, (t, 0, 1))/2
        zero(remainder-a**power)
        checked += 1
    B, M, m, C, eps, E = sp.symbols('B M m C eps E', positive=True)
    amplitude_cap = B*eps**3/(2*C)
    energy_cap = m*B**2*eps**6/(8*C**2)
    zero(2*energy_cap/m-amplitude_cap**2)
    flux_bound = B**2*(E/(2*M))**2/4
    zero(flux_bound-B**2*E**2/(16*M**2))
    c, delta = sp.symbols('c delta', positive=True)
    gain = c*E**2-delta**2*E
    zero((gain-c*E**2/2).subs(E, 2*delta**2/c))
    return dict(integral_remainder_powers=checked, third_derivative_factor='1/6',
                amplitude_cap='B0*epsilon^3/(2*C3)',
                sufficient_energy_cap='m*B0^2*epsilon^6/(8*C3^2)',
                radiation_lower_coefficient='B0^2/(16*M^2)',
                gain_lower_energy='2*delta_L^2/c_rad',
                additional_requirement='Both inverse branches remain inside the physical parameter and energy neighborhoods')


def exact_branch_models():
    rows = []
    for carrier in (64, 80, 96, 128, 160, 192):
        eps, delta = Fraction(1, carrier**2), Fraction(1, 2**carrier)
        c_rad = Fraction(1, 16)
        for shift in (Fraction(1), 1/eps):
            for denominator in (8, 16, 32):
                s = eps**3/denominator
                energy = s*s
                assert 2*delta*delta/c_rad <= energy <= eps**6/64
                amplitudes = []
                for sign in (1, -1):
                    signed_s = sign*s
                    # E(a)=(a/(1-lambda*a))^2 has a nonzero cubic mass term.
                    a = signed_s/(1+shift*signed_s)
                    q = 1/(1-shift*a)**2
                    assert a*a*q == energy
                    assert Fraction(1, 2) <= q <= 2
                    assert abs(a) <= eps**3/2
                    waveform = (delta*a-a*a, a*a-a**3/eps**3)
                    projected = waveform[1]
                    assert abs(projected) >= a*a/2
                    flux = sum(value*value for value in waveform)
                    linear = delta*delta*energy
                    assert flux >= c_rad*energy**2
                    assert flux-linear >= c_rad*energy**2/2
                    assert linear/flux <= delta*delta/(c_rad*energy)
                    amplitudes.append(a)
                assert amplitudes[0] != -amplitudes[1]
                rows.append(dict(carrier=carrier, clock_independent_energy_model=True,
                                 mass_parameter='1' if shift == 1 else 'epsilon^(-1)',
                                 signed_energy_amplitude=f'epsilon^3/{denominator}',
                                 both_branches_checked=True, unequal_amplitude_magnitudes=True))
    return dict(cases=len(rows), individual_branches=2*len(rows), records=rows,
                model='N(a)=(delta*a-a^2, a^2-epsilon^(-3)*a^3)',
                energy_model='E(a)=(a/(1-lambda*a))^2',
                toy_scaling='epsilon=L^(-2), delta=2^(-L)',
                scope='Exact rational norm inequalities, not physical carrier or amplitude thresholds')


def asymptotic_window_and_fraction():
    L, P, N, c = sp.symbols('L P N c', positive=True)
    log_ratio = (2*N+6*P)*sp.log(L)-2*c*L
    zero(sp.limit(log_ratio/L, L, sp.oo)+2*c)
    eps, c2 = sp.symbols('eps c2', positive=True)
    energy = c2*eps**6
    zero(energy*eps**(-4)-c2*eps**2)
    zero(sp.sqrt(energy)*eps**(-1)-sp.sqrt(c2)*eps**2)
    return dict(lower_to_upper_energy_ratio_tends_to_zero=True,
                exact_sequence='E_L=c2*epsilon_L^6',
                outgoing_fraction_upper='C*delta_L^2+C*c2*epsilon_L^2',
                initial_norm_upper='C*sqrt(c2)*epsilon_L^2',
                available_selected_phases=['0', 'pi/2'],
                opposite_sign_phase_selection_needed=False,
                scope='Limits for each fixed finite P,N; no uniform late-time limit is implied')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "clock": projected_clock_jets(),
        "constants": integral_remainder_and_constants(),
        "branches": exact_branch_models(),
        "limits": asymptotic_window_and_fraction(),
    }, indent=2))
