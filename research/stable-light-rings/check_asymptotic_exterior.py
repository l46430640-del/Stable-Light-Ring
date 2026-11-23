"""Validate a nonlinear, asymptotically flat Einstein--scalar exterior.

The proof uses a contraction on the entire half-line, not a boundary
condition at another large radius. It does not yet match this exterior
to a globally validated interior or certify its response derivatives.
"""

import argparse
import hashlib
import json
from math import inf, nextafter
from pathlib import Path
from time import perf_counter

import mpmath
from mpmath import iv
import numpy as np

from check_barrier_certificate import exact, enclosure
from check_finite_static_inverse import upper
from check_regular_center import Dual, exponential, promote
from trapped_axial import Background


def lower(value):
    return nextafter(float(value.a), -inf)


def scaled_kernels(invr, decay2, radial_decay2, state, mass_R, omega,
                   amplitude, mass_scale, lapse_scale, sigma, pi):
    """Factor out a_* exp(-beta x) in the scalar equation and its square
    in the metric integrands. radial_decay2 denotes r exp(-2 beta x).
    """
    U, W, v, h, invr, decay2, radial_decay2, mass_R, omega, amplitude, mass_scale, lapse_scale, sigma, pi = promote(
        [*state, invr, decay2, radial_decay2, mass_R, omega, amplitude, mass_scale, lapse_scale, sigma, pi])
    mass, eta = mass_R+mass_scale*v, lapse_scale*h
    f = 1-2*mass*invr
    ratio = amplitude*amplitude*decay2*U*U/(sigma*sigma)
    potential_ratio = (1-2*ratio)*(1-2*ratio)
    Vs = (1-2*ratio)*(1-6*ratio)
    redshift = exponential(-2*eta)
    P = 2*invr+2*mass*invr*invr/f-4*pi*amplitude*amplitude*radial_decay2*U*U*potential_ratio/f
    Q = (Vs-omega*omega*redshift/f)/f
    residual = -P*W+(Q-(1-omega*omega))*U
    mass_kernel = 2*pi*amplitude*amplitude/mass_scale*(omega*omega*redshift/f*U*U+f*W*W+U*U*potential_ratio)
    lapse_kernel = -4*pi*amplitude*amplitude/lapse_scale*(omega*omega*redshift/(f*f)*U*U+W*W)
    return [residual, mass_kernel, lapse_kernel], [P, Q, f, potential_ratio]


def certify(digits=30):
    iv.dps = digits
    R, beta, amplitude = iv.mpf(40), iv.mpf("0.5"), iv.mpf("1e-8")
    mass_scale, lapse_scale = iv.mpf("1e6")*amplitude*amplitude, iv.mpf("1e5")*amplitude*amplitude
    sigma = iv.mpf("0.08")
    mass_R, omega = enclosure(iv.mpf("5.64"), iv.mpf("5.66")), enclosure(iv.mpf("0.139"), iv.mpf("0.141"))
    invr, decay2, radial_decay2 = enclosure(0, 1/R), enclosure(0, 1), enclosure(0, R)
    box = [enclosure(-3, 3), enclosure(-3, 3), enclosure(-1, 1), enclosure(-1, 1)]
    parameters = [mass_R, omega, amplitude, mass_scale, lapse_scale, sigma, iv.pi]
    kernels, coefficients = scaled_kernels(invr, decay2, radial_decay2, box, *parameters)
    P, Q, f, potential_ratio = coefficients
    kappa = iv.sqrt(1-omega*omega)
    assert lower(kappa-beta) > 0
    # Both half-line Green kernels have this common bound since kappa < 1.
    gain = 1/(kappa*kappa-beta*beta)
    I1 = R/(2*beta)+1/(2*beta)**2
    I2 = R*R/(2*beta)+2*R/(2*beta)**2+2/(2*beta)**3
    image = [upper(1+gain*abs(kernels[0])), upper(I2*abs(kernels[1])), upper(I1*abs(kernels[2]))]
    derivative, _ = scaled_kernels(invr, decay2, radial_decay2,
        [Dual(value, np.eye(4, dtype=int)[j]) for j, value in enumerate(box)], *parameters)
    factors = [gain, I2, I1]
    lipschitz_rows = [upper(factor*sum(map(abs, entry.d), iv.mpf(0))) for factor, entry in zip(factors, derivative)]
    scalar_gain = upper(gain*sum(map(abs, derivative[0].d[:2]), iv.mpf(0)))
    assert exact(image[0]) < 3 and exact(image[1]) < 1 and exact(image[2]) < 1
    assert all(exact(value) < iv.mpf(bound) for value, bound in zip(image, ("2.839", "0.193", "0.049")))
    assert all(exact(value) < iv.mpf(bound) for value, bound in zip(lipschitz_rows, ("0.613", "0.129", "0.033")))
    assert exact(max(lipschitz_rows)) < iv.mpf("0.64")
    assert exact(scalar_gain) < iv.mpf(2)/3
    assert lower(f) > .7169 and upper(kappa) < 1

    # Comparison exponentials give a sharper physical decay rate. Positivity
    # of P uses r^2 exp(-2 beta (r-R)) <= R^2, not independent interval ranges.
    lam, gam = iv.mpf("0.98"), iv.mpf("1.21")
    P_sign = 2-4*iv.pi*R*R*9*amplitude*amplitude*exact(upper(abs(potential_ratio)))/exact(lower(f))
    low_residual = lam*lam-Q
    high_residual = gam*gam-exact(upper(P))*gam-exact(upper(Q))
    assert lower(mass_R-mass_scale) > 0 and lower(P_sign) > 0
    assert upper(low_residual) < 0 and lower(high_residual) > 0
    I1sharp = R/(2*lam)+1/(2*lam)**2
    I2sharp = R*R/(2*lam)+2*R/(2*lam)**2+2/(2*lam)**3
    fmin = exact(lower(f))
    redshift_max = iv.exp(2*lapse_scale)
    mass_coefficient = upper(2*iv.pi*I2sharp*(omega*omega*redshift_max/fmin+gam*gam+exact(upper(abs(potential_ratio)))))
    lapse_coefficient = upper(4*iv.pi*I1sharp*(omega*omega*redshift_max/(fmin*fmin)+gam*gam))
    assert exact(mass_coefficient) < 14000 and exact(lapse_coefficient) < 400
    return {
        "parameters": {"R": 40, "sigma": "0.08", "mu": 1, "amplitude_absolute_upper": "1e-8",
                       "mass_R_interval": ["5.64", "5.66"], "omega_interval": ["0.139", "0.141"]},
        "weights": {"exponential_rate": "0.5", "scalar_scale": "1e-8", "mass_scale": "1e-10", "log_z_scale": "1e-11"},
        "bound_row_order": ["common_bound_for_scalar_and_its_derivative", "mass", "log_z"],
        "image_bounds": image, "contraction_row_bounds": lipschitz_rows,
        "scalar_block_contraction_bound": scalar_gain, "metric_factor_lower_bound": lower(f),
        "comparison_bounds": {"r_times_P_lower_bound": lower(P_sign), "Q_lower": lower(Q), "P_upper": upper(P), "Q_upper": upper(Q),
                              "slow_exponential_residual_upper": upper(low_residual), "fast_exponential_residual_lower": lower(high_residual)},
        "quadratic_tail_coefficients": {"mass_upper": mass_coefficient, "minus_log_z_upper": lapse_coefficient},
        "verified_decimal_bounds": {"contraction": "0.64", "decay_rate_lower": "0.98", "logarithmic_slope_upper": "1.21",
                                    "mass_tail_coefficient": "14000", "minus_log_z_coefficient": "400"},
        "arithmetic": f"{digits}-digit intervals, analytic Green integrals over the entire half-line"}


def endpoint_enclosure(background):
    radius = exact(float(background.radius))
    argument = radius-exact(float(background.sol.x[-2]))
    values = []
    for j in range(4):
        value = iv.mpf(0)
        for coefficient in background.sol.c[:, -1, j]:
            value = value*argument+exact(float(coefficient))
        values.append(value)
    return values


def compare_saved_endpoint(background):
    a, b, u, zeta = endpoint_enclosure(background)
    R, A0 = exact(float(background.radius)), iv.mpf(str(background.central))
    assert background.radius == 40 and str(background.scale) == "0.08"
    A_R, M_R = A0*a, R*u
    omega = iv.exp(zeta)*iv.sqrt(1-2*u)
    assert lower(A_R) > 0 and upper(A_R) < 1e-8
    assert lower(M_R) > 5.64 and upper(M_R) < 5.66
    assert lower(omega) > .139 and upper(omega) < .141
    exact_b = enclosure(-iv.mpf("1.21"), -iv.mpf("0.98"))*a
    b_error = upper(abs(exact_b-b))
    mass_tail = upper(14000*A_R*A_R)
    zeta_error = upper(400*A_R*A_R)
    assert exact(b_error) < iv.mpf("3e-15")
    assert exact(mass_tail) < iv.mpf("1.4e-26")
    assert exact(zeta_error) < iv.mpf("4e-28")
    return {"amplitude_R_interval": [lower(A_R), upper(A_R)],
            "mass_R_interval": [lower(M_R), upper(M_R)], "omega_interval": [lower(omega), upper(omega)],
            "normalized_scalar_derivative_error_bound": b_error, "local_log_frequency_error_bound": zeta_error,
            "ADM_mass_minus_mass_R_bound": mass_tail,
            "scope": "The exact exterior matches represented A(R), m(R) and has omega=exp(zeta_hat(R))*sqrt(f_hat(R)). Its derivative and zeta(R) have the displayed errors."}


def check_formulas():
    from boson_star import equations, SINGULAR
    from scipy.integrate import quad
    iv.dps = 30
    # Exercise the nonlinear terms at a larger amplitude than the certified
    # tail; tiny physical tails would conceal missing factors in these tests.
    r, beta, amplitude = 42., .5, .003
    mass_scale, lapse_scale = .01, .001
    X = np.array([.5, -.6, .1, -.02])
    M, omega = 5.65, .14
    E = np.exp(-beta*(r-40))
    args = [1/r, E*E, r*E*E]
    parameters = [M, omega, amplitude, mass_scale, lapse_scale, .08, np.pi]
    derivative, _ = scaled_kernels(*args, [Dual(v, np.eye(4, dtype=int)[j]) for j, v in enumerate(X)], *parameters)
    for j in range(4):
        shifted = X.astype(complex)
        shifted[j] += 1e-25j
        result, _ = scaled_kernels(*args, shifted, *parameters)
        expected = np.imag(result)/1e-25
        actual = np.array([float(entry.d[j].mid) for entry in derivative])
        assert np.max(np.abs(actual-expected)) < 1e-12
    kernels, _ = scaled_kernels(*args, X, *parameters)
    A, B = amplitude*E*X[:2]
    mass, eta = M+mass_scale*X[2], lapse_scale*X[3]
    f = 1-2*mass/r
    y = np.array([A/.06, B/.06, mass/r, np.log(omega)-eta-.5*np.log(f)])
    mass_prime = mass_scale*r*r*E*E*kernels[1]
    eta_prime = -lapse_scale*r*E*E*kernels[2]
    B_prime = amplitude*E*kernels[0]+(1-omega*omega)*A
    rebuilt = np.array([B/.06, B_prime/.06, mass_prime/r-mass/(r*r), -eta_prime+mass_prime/(r*f)-mass/(r*r*f)])
    expected = equations(np.array([r]), y[:, None], .06, .08)[:, 0]+SINGULAR@y/r
    assert np.max(np.abs(rebuilt-expected)/np.array([amplitude, amplitude, 1., 1.])) < 1e-12
    zeros, _ = scaled_kernels(*args, [0., 0., 0., 0.], *parameters)
    assert np.all(np.array(zeros) == 0)
    kappa = np.sqrt(1-omega*omega)
    for x in (.1, 1., 10.):
        def integrand(t):
            return -(np.exp(-kappa*abs(x-t))-np.exp(-kappa*(x+t)))/(2*kappa)*np.exp(-beta*t)
        computed = quad(integrand, 0, x, epsabs=1e-13)[0]+quad(integrand, x, np.inf, epsabs=1e-13)[0]
        analytic = -(np.exp(-beta*x)-np.exp(-kappa*x))/(kappa*kappa-beta*beta)
        assert abs(computed-analytic) < 1e-12
        def derivative_integrand(t):
            return (np.sign(x-t)*np.exp(-kappa*abs(x-t))-np.exp(-kappa*(x+t)))/2*np.exp(-beta*t)
        derivative_value = quad(derivative_integrand, 0, x, epsabs=1e-13)[0]+quad(derivative_integrand, x, np.inf, epsabs=1e-13)[0]
        analytic_derivative = (beta*np.exp(-beta*x)-kappa*np.exp(-kappa*x))/(kappa*kappa-beta*beta)
        assert abs(derivative_value-analytic_derivative) < 1e-12
    print("PASS: interval derivatives, original field equations, vacuum exterior, half-line Green kernel", flush=True)


def check_finite_exterior(background, amplitude=1e-8):
    """A finite-domain numerical comparison, separate from the half-line proof."""
    from scipy.integrate import solve_bvp
    R, outer, M, omega, amp = 40., 80., background.mass, background.omega, float(amplitude)
    assert 0 < amp <= 1e-8
    mesh = np.linspace(R, outer, 401)
    guess_a = np.exp(-1.15*(mesh-R))
    guess = np.array([guess_a, -1.15*guess_a, .014*(1-guess_a*guess_a), -.004*guess_a*guess_a])
    def rhs(r, y):
        a, b, v, h = y
        mass, eta = M+1e6*amp*amp*v, 1e5*amp*amp*h
        f = 1-2*mass/r
        ratio = amp*amp*a*a/.08**2
        potential_ratio = (1-2*ratio)**2
        Vs = (1-2*ratio)*(1-6*ratio)
        Omega2 = omega*omega*np.exp(-2*eta)/f
        P = 2/r+2*mass/(r*r*f)-4*np.pi*r*amp*amp*a*a*potential_ratio/f
        Q = (Vs-Omega2)/f
        return np.array([b, -P*b+Q*a,
                         2*np.pi*r*r/1e6*(Omega2*a*a+f*b*b+a*a*potential_ratio),
                         4*np.pi*r/1e5*(Omega2/f*a*a+b*b)])
    def boundary(left, right):
        mass = M+1e6*amp*amp*right[2]
        kappa = np.sqrt(1-omega*omega)
        power = mass*(2*omega*omega-1)/kappa
        return np.array([left[0]-1, left[2], right[1]+(kappa+(1-power)/outer)*right[0], right[3]])
    solution = solve_bvp(rhs, boundary, mesh, guess, tol=1e-8, max_nodes=4000)
    if not solution.success:
        raise RuntimeError("Exterior consistency calculation reached its resource/solver limit: "+solution.message)
    left, right = solution.sol(R), solution.sol(outer)
    assert .98 < -left[1] < 1.21 and 0 < 1e6*right[2] < 14000 and 0 < -1e5*left[3] < 400
    reference = background.sol(R)
    reference_slope = -reference[1]/reference[0]
    return {"radial_nodes": int(len(solution.x)), "outer_radius": outer,
            "minus_Aprime_over_A_at_R": float(-left[1]), "reference_logarithmic_slope_at_R": float(reference_slope),
            "logarithmic_slope_difference": float(-left[1]-reference_slope),
            "mass_tail_over_A_R_squared": float(1e6*right[2]),
            "minus_log_z_R_over_A_R_squared": float(-1e5*left[3]),
            "scope": "Floating-point finite-domain consistency only; not used by the half-line existence proof."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert args.digits >= 25
    start = perf_counter()
    check_formulas()
    background = Background()
    result = {"scope": "Exact nonlinear asymptotically flat exterior family; this calculation does not certify global interior matching or response derivatives.",
              "exterior_certificate": certify(args.digits), "represented_endpoint_comparison": compare_saved_endpoint(background),
              "finite_domain_crosscheck": check_finite_exterior(background), "mpmath_version": mpmath.__version__,
              "background_sha256": hashlib.sha256((Path(__file__).parent/"data/s08a06/background.npz").read_bytes()).hexdigest(),
              "elapsed_seconds": perf_counter()-start}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
