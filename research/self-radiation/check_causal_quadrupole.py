"""Algebraic checks: causal quadrupole."""
import json
import sympy as sp


def complex_chart_checks():
    f_lower, mass_upper = sp.Rational(7169, 10000), sp.Rational(566, 100)
    # The inherited lapse correction is below 10^-11; exp(2e-11) < 1+3e-11.
    omega2 = sp.Rational(141, 1000)**2*(1+sp.Rational(3, 10**11))/f_lower
    radial_pressure_factor = (omega2+sp.Rational(121, 100)**2+1)/2
    rn_bound = (mass_upper/40+4*sp.Rational(22, 7)*1600*sp.Rational(1, 10**16)
                *radial_pressure_factor)/f_lower
    assert rn_bound < sp.Rational(1, 5)
    assert 2*(1+rn_bound)**2 < sp.Rational(288, 100)
    assert 10-sp.Rational(22304, 10**17)-sp.Rational(288, 100) > 0
    z, a, b, w = sp.symbols('z a b w')
    polynomial = z*z-a*z+b
    difference = sp.expand(polynomial.subs(z, w*z)-polynomial)
    assert sp.expand(difference-((w*w-1)*z*z-a*(w-1)*z)) == 0
    # |a| <= 10 and z/(1+z^2) <= 1/2 for real z >= 0.
    assert sp.factor(1+z*z-2*z) == (z-1)**2
    delta = sp.Rational(1, 100)
    error = 7*delta+delta*delta
    margin = sp.Rational(1, 10)-error
    assert error == sp.Rational(701, 10000)
    assert margin == sp.Rational(299, 10000) and margin > 0
    # Permit slight spatial growth so varying nonzero frequencies are continuous.
    tail_rate, allowance, imaginary_part = sp.Rational(3, 4), sp.Rational(1, 16), sp.Rational(1, 16)
    assert tail_rate-allowance-imaginary_part > sp.Rational(1, 2)
    return dict(absolute_rn_upper=float(rn_bound), relative_squared_frequency_radius=str(delta),
                relative_determinant_error=str(error), complex_determinant_margin=str(margin),
                tail_exponent_after_allowances=str(tail_rate-allowance-imaginary_part),
                scope='Uniform algebraic chart bound; no numerical matching radius in frequency.')


def initial_tail_checks():
    rho, M, Q = sp.symbols('rho M Q', positive=True)
    psi = 1+M/(2*rho)
    radius = rho*psi**2
    f = (1-M/(2*rho))**2/psi**2
    z = Q/(psi*rho**3)
    dr = lambda expression: sp.diff(expression, rho)/sp.diff(radius, rho)
    radial_laplacian = psi**-4*(sp.diff(z, rho, 2)
        +(2/rho+2*sp.diff(psi, rho)/psi)*sp.diff(z, rho)-6*z/rho**2)
    assert sp.factor(radial_laplacian) == 0
    master = sp.factor(radius/3*(4*z+f/(2+3*M/radius)*(4*z-4*radius*dr(z))))
    expected = Q*(M+2*rho)**2*(M+12*rho)/(3*rho**3*(M*M+10*M*rho+4*rho*rho))
    assert sp.factor(master-expected) == 0
    potential = f*(24*radius**3+24*M*radius**2+36*M*M*radius+18*M**3)
    potential /= radius**3*(2*radius+3*M)**2
    acceleration = sp.factor(f*dr(f*dr(master))-potential*master)
    expected_acceleration = -128*M*Q*(2*rho-M)*(rho+2*M)
    expected_acceleration /= (M+2*rho)**5*(M*M+10*M*rho+4*rho*rho)
    assert sp.factor(acceleration-expected_acceleration) == 0
    assert sp.limit(rho**2*master, rho, sp.oo) == 4*Q
    assert sp.limit(rho**5*acceleration, rho, sp.oo) == -2*M*Q
    assert acceleration.subs(M, 0) == 0
    assert sp.factor(sp.diff(z, rho)) != 0
    # The tensor is nonstationary already without the slow modulation.
    angle, phase, eta = sp.symbols('angle phase eta', real=True)
    p2 = (3*sp.Rational(3, 4)*sp.cos(angle)**2-1)/2
    density = 1+eta*sp.cos(2*angle-phase)
    averaged = sp.integrate(sp.expand_trig(p2*density), (angle, 0, 2*sp.pi))/(2*sp.pi)
    assert sp.trigsimp(averaged-sp.Rational(1, 16)-sp.Rational(9, 32)*eta*sp.cos(phase)) == 0
    assert sp.Rational(1, 16)/sp.Rational(9, 32) == sp.Rational(2, 9)
    return dict(vacuum_constraint_laplacian=True, moncrief_initial_field=str(master),
                exact_initial_acceleration=str(acceleration), leading_acceleration='-2*M*Q/rho^5',
                positive_density_modulation_limit='2/9', zero_mass_static_limit=True,
                weighted_radial_L2_range='s < 3/2',
                scope='Exact Schwarzschild comparison tail; stellar deviations are exponentially small analytically.')


def causal_residue_checks():
    frequency, nu, A, density = sp.symbols('frequency nu A density', nonzero=True)
    positive_wave_transform = sp.I*A/(2*(frequency-nu))
    news_transform = -sp.I*frequency*positive_wave_transform
    assert sp.limit((frequency-nu)*news_transform, frequency, nu) == nu*A/2
    transformed_density = sp.I*density/(frequency-nu)
    divergence_current = sp.I*nu*transformed_density
    assert sp.cancel(-sp.I*frequency*transformed_density+divergence_current-density) == 0
    # An unstable oscillator still has an uncancellable forcing residue when
    # its initial-data transform is regular at the forcing frequency.
    s, rate, q0, v0, force = sp.symbols('s rate q0 v0 force')
    transform = (s*q0+v0+force/(s+sp.I*nu))/(s*s-rate*rate)
    residue = sp.simplify(sp.limit((s+sp.I*nu)*transform, s, -sp.I*nu))
    assert sp.factor(residue+force/(nu*nu+rate*rate)) == 0
    assert not residue.has(q0, v0)
    x, R, k = sp.symbols('x R k', positive=True)
    energy = sp.integrate(k*k*sp.sin(k*x)**2, (x, 0, R))
    assert sp.simplify(energy-k*k*R/2+k*sp.sin(2*k*R)/4) == 0
    assert sp.limit(energy/R, R, sp.oo) == k*k/2
    epsilon = sp.Symbol('epsilon', positive=True)
    assert sp.limit(sp.exp(epsilon*x)/x**2, x, sp.oo) == sp.oo
    return dict(real_wave_news_boundary_residue='nu*Aout/2',
                causal_conservation_initial_term=True,
                unstable_oscillator_residue=str(residue),
                periodic_initial_energy_per_radius='nu^2*abs(Aout)^2/2',
                polynomial_tail_blocks_naive_complex_Jost_integral=True,
                scope='Residue algebra and obstruction examples; stellar initial-news regularity is unproved.')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "complex_chart": complex_chart_checks(),
        "initial_tail": initial_tail_checks(),
        "causal_residue": causal_residue_checks(),
    }, indent=2))
