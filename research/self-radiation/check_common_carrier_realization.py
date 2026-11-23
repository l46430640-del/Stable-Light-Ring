"""Algebraic checks: common carrier realization."""
from fractions import Fraction
import json
from math import comb
import sympy as sp
from sympy.physics.wigner import wigner_d_small


def zero(value):
    if isinstance(value, sp.MatrixBase):
        assert value.applyfunc(sp.simplify) == sp.zeros(*value.shape)
    else:
        assert sp.simplify(value) == 0


def five_plane_geometry():
    c, s, lam = sp.symbols('c s lam', real=True)
    z0 = sp.Matrix([sp.Rational(1, 2), sp.I, -sp.sqrt(3)/2])
    z1 = sp.Matrix([c/2-sp.I*s, s/2+sp.I*c, -sp.sqrt(3)/2])
    dot = (1+3*c)/4
    same, reverse = z0.dot(sp.conjugate(z1))/2, z0.dot(z1)/2
    zero(sp.expand(same*sp.conjugate(same)-((1+dot)/2)**2).subs(s*s, 1-c*c))
    zero(reverse-(1-dot)/2)
    zero(z0.dot(z0))
    zero(z0.dot(sp.conjugate(z0))-2)
    values = [sp.simplify(dot.subs(c, (sp.sqrt(5)-1)/4)),
              sp.simplify(dot.subs(c, -(sp.sqrt(5)+1)/4))]
    maximum = (1+3*sp.sqrt(5))/16
    rho = (17+3*sp.sqrt(5))/32
    zero(values[0]-maximum)
    assert 0 < -values[1] < maximum < 1
    assert rho < sp.Rational(3, 4)
    matrix = sp.Matrix([[1+c*c, c*s, 0], [c*s, s*s, 0], [0, 0, 0]])
    characteristic = sp.expand((lam*sp.eye(3)-matrix).det()).subs(s*s, 1-c*c)
    zero(characteristic-lam*((lam-1)**2-c*c))
    return dict(pairwise_normal_products=[str(v) for v in values],
                absolute_overlap_bound=str(rho), plane_gap=str(1-maximum),
                positive_form_eigenvalues=['0', '1-c', '1+c'],
                normalized_complex_null_vectors=True,
                scope='Exact five-plane geometry and highest-weight contraction factors')


def angular_norm(degree):
    total = 0
    for m in range(-degree, degree+1):
        if m % 5 == 0:
            sign = 1 if m % 2 == 0 else -1
            total += comb(2*degree, degree+m)*(3**(degree+m)+3**(degree-m)-2*sign*3**degree)
    return Fraction(5*total, 2*16**degree)


def exact_angular_normalizations():
    independent = []
    for degree in (1, 3, 5, 7):
        rotation = wigner_d_small(degree, sp.pi/3)
        norm = sp.Rational(5, 2)*sum(
            (rotation[degree-m, 0]-sp.Integer(-1)**m*rotation[degree+m, 0])**2
            for m in range(-degree, degree+1) if m % 5 == 0)
        expected = angular_norm(degree)
        zero(norm-sp.Rational(expected.numerator, expected.denominator))
        independent.append(dict(degree=degree, norm_squared=str(expected)))
    checks = 0
    largest_relative_bound = Fraction(0)
    for degree in range(11, 102, 2):
        norm = angular_norm(degree)
        error = 8*Fraction(3, 4)**degree
        assert abs(norm-1) <= error < Fraction(1, 2)
        assert Fraction(1, 2) < norm < Fraction(3, 2)
        largest_relative_bound = max(largest_relative_bound, abs(norm-1)/error)
        checks += 1
    assert angular_norm(1) == angular_norm(3) == 0
    # This common normalization fixes the phase mean, without altering a1/a0.
    amplitude0, amplitude1, energy = Fraction(1), Fraction(1, 10), Fraction(1)
    eta = 2*amplitude0*amplitude1/(amplitude0**2+amplitude1**2)
    assert 0 < eta < Fraction(2, 9)
    for degree in (11, 21, 41, 81):
        denominator = amplitude0**2*angular_norm(degree)+amplitude1**2*angular_norm(degree+2)
        normalization_squared = energy/denominator
        assert normalization_squared*denominator == energy
        assert Fraction(1, 2)*(amplitude0**2+amplitude1**2) < denominator
    return dict(independent_wigner_checks=independent, odd_degree_bound_checks=checks,
                largest_fraction_of_bound=float(largest_relative_bound),
                exact_low_degree_cancellations=[1, 3], example_modulation=str(eta),
                common_energy_normalization_checks=4,
                scope='Angular norms of the specified wave sum, not stellar radial modes')


def source_channels_and_phase():
    modes = [(node, degree, direction) for node in range(5)
             for degree in (0, 2) for direction in (-1, 1)]
    counts = dict(same_plane_slow=0, same_plane_high_magnetic=0, different_planes=0)
    ell = 11
    differences = set()
    for ni, di, si in modes:
        for nj, dj, sj in modes:
            if ni != nj:
                counts['different_planes'] += 1
            elif si == sj:
                assert abs(si*(ell+di)-sj*(ell+dj)) <= 2
                differences.add((di-dj)//2)
                counts['same_plane_slow'] += 1
            else:
                assert abs(si*(ell+di)-sj*(ell+dj)) > 2
                counts['same_plane_high_magnetic'] += 1
    assert len(modes) == 20 and sum(counts.values()) == 400
    assert differences == {-1, 0, 1}
    z = sp.symbols('z')
    cyclotomic = sp.Poly(sum(z**j for j in range(5)), z)
    for m in (-2, -1, 1, 2):
        polynomial = sp.Poly(sum(z**((m*j) % 5) for j in range(5)), z)
        assert sp.rem(polynomial, cyclotomic).is_zero
    for charge in (-2, 2):
        zero((1+sp.I**charge)/2)
    u, v, U, V = sp.symbols('u v U V', real=True)
    complex_product = (u+sp.I*v)*(U-sp.I*V)
    zero((u*U+v*V)/2-sp.re(complex_product)/2)
    t, theta, omega, F = sp.symbols('t theta omega F', real=True)
    real_master = sp.re(sp.exp(sp.I*theta)*sp.I*F*sp.exp(-sp.I*omega*t)).expand(complex=True)
    zero(real_master.subs(t, 0)+F*sp.sin(theta))
    zero(sp.diff(real_master, t).subs(t, 0)-omega*F*sp.cos(theta))
    return dict(carriers=len(modes), ordered_difference_products=counts,
                difference_frequency_multipliers=sorted(differences),
                quadrupole_magnetic_indices_after_fivefold_symmetry=[0],
                common_two_phase_filter_retains_beats=True,
                master_initial_data=['-F*sin(theta)', 'omega*F*cos(theta)'],
                scope='Magnetic selection in each common plane; distinct-plane sources use the analytic overlap bound')


def cross_plane_cap_budget():
    d = sp.symbols('d', nonnegative=True, integer=True)
    assert sp.expand(4*(d+1)**3-(2*d+1)**2*(d+2)-(3*d+2)) == 0
    z, A = sp.symbols('z A', positive=True)
    zero(1-z*z-(1-z)-z*(1-z))
    zero(sp.integrate(sp.exp(-A*(1-z)), (z, 0, 1))-(1-sp.exp(-A))/A)
    allocations = 0
    for p in range(7):
        for q in range(7):
            for ell in (max(2*p+2, 2*q+2, 11), max(2*p+2, 2*q+2, 11)+2):
                assert ell-p >= ell/2 and ell+2-q >= (ell+2)/2
                allocations += 1
    return dict(normalization_induction_positive_remainder='3*d+2',
                normalized_derivative_allocations=allocations,
                sphere_cap_upper='4*pi/A', angular_product_L1_order='L^(-1/2)',
                normalized_angular_energy_error='O(rho_ang^ell), rho_ang<3/4',
                scope='The source L1 bound uses absolute products, distinct from exponentially small complex energy inner products')


if __name__ == "__main__":
    import json
    if not __debug__:
        raise SystemExit("Run without -O.")
    print(json.dumps({
        "geometry": five_plane_geometry(),
        "normalization": exact_angular_normalizations(),
        "channels": source_channels_and_phase(),
        "caps": cross_plane_cap_budget(),
    }, indent=2))
