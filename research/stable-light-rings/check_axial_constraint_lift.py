"""Exact algebra for fixed-charge gravitational initial data on a static star.

The existence argument uses the weighted Laplacian theorem, not these checks.
No equilibrium, boundary-value problem, or time evolution is solved here.
"""

import json
import time

import sympy as sp


CHECKS = []
ANGLES = sp.symbols("theta phi", real=True)
TRIG_VARIABLES = sp.symbols("ct st cp sp")
TRIG_RULES = dict(zip(
    (sp.cos(ANGLES[0]), sp.sin(ANGLES[0]), sp.cos(ANGLES[1]), sp.sin(ANGLES[1])),
    TRIG_VARIABLES,
))
TRIG_RELATIONS = [TRIG_VARIABLES[0]**2 + TRIG_VARIABLES[1]**2 - 1,
                  TRIG_VARIABLES[2]**2 + TRIG_VARIABLES[3]**2 - 1]


def trig_polynomial(expression):
    expression = sp.expand_trig(expression)
    expression = expression.subs({sp.cot(t): sp.cos(t) / sp.sin(t) for t in ANGLES})
    expression = expression.subs({sp.tan(t): sp.sin(t) / sp.cos(t) for t in ANGLES})
    numerator, denominator = sp.fraction(sp.cancel(expression.xreplace(TRIG_RULES)))
    numerator = sp.reduced(numerator, TRIG_RELATIONS, *TRIG_VARIABLES, domain="EX")[1]
    return sp.cancel(numerator / denominator)


def zero(name, expression):
    residual = trig_polynomial(expression)
    if residual != 0:
        raise ValueError(f"{name}: nonzero residual {residual}")
    CHECKS.append(name)


def christoffel(metric, coordinates):
    inverse = metric.inv()
    n = len(coordinates)
    return [
        [
            [
                sp.simplify(sum(
                    inverse[a, d] * (
                        sp.diff(metric[d, c], coordinates[b])
                        + sp.diff(metric[d, b], coordinates[c])
                        - sp.diff(metric[b, c], coordinates[d])
                    ) / 2 for d in range(n)
                ))
                for c in range(n)
            ]
            for b in range(n)
        ]
        for a in range(n)
    ]


def divergence_covariant(tensor, inverse, connection, coordinates):
    n = len(coordinates)
    return [
        sum(
            inverse[i, k] * (
                sp.diff(tensor[i, j], coordinates[k])
                - sum(connection[a][k][i] * tensor[a, j]
                      + connection[a][k][j] * tensor[i, a]
                      for a in range(n))
            ) for i in range(n) for k in range(n)
        ) for j in range(n)
    ]


def check_conformal_equation():
    psi, eta = sp.symbols("psi eta", positive=True)
    W, G, V, norm, lap, R = sp.symbols("W G V norm lap R", real=True)
    physical_density = (W * psi**-12 + G * psi**-4 + V) / 2
    constraint = (
        psi**-5 * (-8 * lap + R * psi)
        - eta**2 * psi**-12 * norm - 16 * sp.pi * physical_density
    )
    lichnerowicz = (
        -8 * lap + (R - 8 * sp.pi * G) * psi
        - (eta**2 * norm + 8 * sp.pi * W) * psi**-7
        - 8 * sp.pi * V * psi**5
    )
    zero("Hamiltonian conformal powers", psi**5 * constraint - lichnerowicz)
    algebraic = lichnerowicz.subs({R: 8 * sp.pi * (W + G + V), lap: 0})
    zero("background Hamiltonian", algebraic.subs({psi: 1, eta: 0}))
    zero("linearized potential", sp.diff(algebraic, psi).subs({psi: 1, eta: 0})
         - (64 * sp.pi * W - 32 * sp.pi * V))
    zero("quadratic source coefficient", sp.diff(algebraic, eta, 2).subs(
        {psi: 1, eta: 0}) / 2 + norm)
    density = (W + G + V) / 2
    spatial_trace = (3 * W - G - 3 * V) / 2
    zero("static lapse source", 4 * sp.pi * (density + spatial_trace)
         - (8 * sp.pi * W - 4 * sp.pi * V))

    remainder_w = psi**-7 - 1 + 7 * (psi - 1)
    remainder_v = psi**5 - 1 - 5 * (psi - 1)
    source = (eta**2 * norm * psi**-7 + 8 * sp.pi * W * remainder_w
              + 8 * sp.pi * V * remainder_v)
    zero("exact nonlinear remainder", algebraic
         - ((64 * sp.pi * W - 32 * sp.pi * V) * (psi - 1) - source))
    zero("negative-power convexity", sp.diff(remainder_w, psi, 2) - 56 * psi**-9)
    zero("positive-power convexity", sp.diff(remainder_v, psi, 2) - 20 * psi**3)
    for name, remainder in [("W", remainder_w), ("V", remainder_v)]:
        zero(f"{name} remainder value", remainder.subs(psi, 1))
        zero(f"{name} remainder slope", sp.diff(remainder, psi).subs(psi, 1))

    A, omega, alpha, derivative = sp.symbols("A omega alpha derivative", real=True)
    pi0 = -sp.I * omega * A / alpha
    zero("charge density times volume", -sp.im(A * psi**-6 * pi0) * psi**6
         - omega * A**2 / alpha)
    zero("scalar momentum vanishes", sp.re(psi**-6 * pi0 * derivative))

    # Check the connection change at an arbitrary orthonormal frame point.
    a = sp.symbols("a0:3", real=True)
    s00, s11, s01, s02, s12 = sp.symbols("s00 s11 s01 s02 s12", real=True)
    sigma = sp.Matrix([[s00, s01, s02], [s01, s11, s12],
                       [s02, s12, -s00 - s11]])

    def change(l, i, j):
        return 2 * ((1 if l == i else 0) * a[j]
                    + (1 if l == j else 0) * a[i]
                    - (1 if i == j else 0) * a[l])

    for j in range(3):
        correction = -2 * sum(a[i] * sigma[i, j] for i in range(3))
        correction -= sum(
            change(l, i, i) * sigma[l, j] + change(l, i, j) * sigma[i, l]
            for i in range(3) for l in range(3)
        )
        zero(f"conformal covariant divergence {j}", correction)


def check_ground_state_and_mass():
    r = sp.symbols("r", positive=True)
    theta, phi = sp.symbols("theta phi", real=True)
    f, alpha = sp.Function("f")(r), sp.Function("alpha")(r)
    v = sp.Function("v")(r, theta, phi)

    def laplacian(value):
        return (f * sp.diff(value, r, 2)
                + (2 * f / r + sp.diff(f, r) / 2) * sp.diff(value, r)
                + (sp.diff(value, theta, 2) + sp.cot(theta) * sp.diff(value, theta)
                   + sp.diff(value, phi, 2) / sp.sin(theta)**2) / r**2)

    L_alpha_v = -8 * laplacian(alpha * v) + 8 * laplacian(alpha) * v
    divergence = alpha**2 * laplacian(v) + 2 * alpha * f * sp.diff(alpha, r) * sp.diff(v, r)
    zero("lapse ground-state transform", L_alpha_v + 8 * divergence / alpha)
    conformal_laplacian = alpha**-4 * (
        laplacian(v) + 2 * f * sp.diff(alpha, r) * sp.diff(v, r) / alpha)
    zero("conformal metric Laplacian", L_alpha_v + 8 * alpha**5 * conformal_laplacian)

    a, w, da2, dw2, cross, lap_a = sp.symbols("a w da2 dw2 cross lap_a", nonzero=True)
    weighted_gradient = dw2 - 2 * w * cross / a + w**2 * da2 / a**2
    boundary_divergence = 2 * w * cross / a - w**2 * da2 / a**2 + w**2 * lap_a / a
    zero("ground-state quadratic form boundary term", dw2 + w**2 * lap_a / a
         - weighted_gradient - boundary_divergence)

    M, c, d = sp.symbols("M c d", real=True)
    asymptotic_alpha, w = 1 - M / r, c / r + d / r**2
    flux = -32 * sp.pi * r**2 * sp.sqrt(1 - 2 * M / r) * (
        asymptotic_alpha * sp.diff(w, r) - w * sp.diff(asymptotic_alpha, r))
    zero("lapse-weighted boundary flux", sp.limit(flux, r, sp.oo) - 32 * sp.pi * c)
    radial_a = (1 + c / r)**4
    radial_b = radial_a * (1 / (1 - 2 * M / r) - 1)
    mass = sp.limit(r**2 * (-sp.diff(radial_a, r) + radial_b / r) / 2, r, sp.oo)
    zero("ADM conformal mass coefficient", mass - M - 2 * c)
    zero("ADM energy normalization", 2 * c - (32 * sp.pi * c) / (16 * sp.pi))


def check_axial_tensors():
    r = sp.symbols("r", positive=True)
    theta, phi = sp.symbols("theta phi", real=True)
    coordinates = (r, theta, phi)
    f, u = sp.Function("f")(r), sp.Function("u")(r)
    metric = sp.diag(1 / f, r**2, r**2 * sp.sin(theta)**2)
    inverse = metric.inv()
    connection = christoffel(metric, coordinates)
    sphere = sp.diag(1, sp.sin(theta)**2)
    sphere_inverse = sphere.inv()
    sphere_connection = christoffel(sphere, (theta, phi))
    harmonics = [
        (2, "l2m0", (3 * sp.cos(theta)**2 - 1) / 2),
        (2, "l2m1", sp.sin(theta) * sp.cos(theta) * sp.cos(phi)),
        (3, "l3m2", sp.sin(theta)**2 * sp.cos(theta) * sp.cos(2 * phi)),
    ]
    for ell, name, Y in harmonics:
        angular_lambda = ell * (ell + 1) - 2
        X = sp.Matrix([sp.diff(Y, phi) / sp.sin(theta), -sp.sin(theta) * sp.diff(Y, theta)])
        angular = (theta, phi)
        DX = sp.Matrix(2, 2, lambda a, b: sp.diff(X[b], angular[a])
                       - sum(sphere_connection[c][a][b] * X[c] for c in range(2)))
        X2 = (DX + DX.T) / 2
        zero(f"{name} vector divergence", sum(sphere_inverse[a, b] * DX[a, b]
                                               for a in range(2) for b in range(2)))
        div_sphere = divergence_covariant(X2, sphere_inverse, sphere_connection, angular)
        for j in range(2):
            zero(f"{name} tensor harmonic divergence {j}", div_sphere[j]
                 + sp.Rational(angular_lambda, 2) * X[j])
        v = 2 * r**2 * f / angular_lambda * (
            sp.diff(u, r) + (2 / r + sp.diff(f, r) / (2 * f)) * u)
        sigma = sp.zeros(3)
        for a in range(2):
            sigma[0, a + 1] = sigma[a + 1, 0] = u * X[a]
            for b in range(2):
                sigma[a + 1, b + 1] = v * X2[a, b]
        zero(f"{name} spatial trace", sp.trace(inverse * sigma))
        for j, div in enumerate(divergence_covariant(sigma, inverse, connection, coordinates)):
            zero(f"{name} spatial divergence {j}", div)
        norm = sp.trace(inverse * sigma * inverse * sigma)
        vector_norm = (X.T * sphere_inverse * X)[0]
        tensor_norm = sp.trace(sphere_inverse * X2 * sphere_inverse * X2)
        zero(f"{name} spatial norm", norm - 2 * f * u**2 * vector_norm / r**2
             - v**2 * tensor_norm / r**4)
        # Exact beta moments avoid an unbounded general trigonometric integrator.
        def integrate(value):
            polynomial = sp.Poly(trig_polynomial(value), *TRIG_VARIABLES, domain="EX")
            total = 0
            for (ct, st, cp, ss), coefficient in polynomial.terms():
                if ct % 2 or cp % 2 or ss % 2:
                    continue
                beta = lambda a, b: sp.gamma(a) * sp.gamma(b) / sp.gamma(a + b)
                total += coefficient * beta(sp.Rational(st + 2, 2), sp.Rational(ct + 1, 2)) * (
                    2 * beta(sp.Rational(ss + 1, 2), sp.Rational(cp + 1, 2)))
            return sp.simplify(total)
        scalar_integral = integrate(Y**2)
        zero(f"{name} vector norm integral", integrate(vector_norm)
             - ell * (ell + 1) * scalar_integral)
        zero(f"{name} tensor norm integral", integrate(tensor_norm)
             - sp.Rational(ell * (ell + 1) * angular_lambda, 2) * scalar_integral)

    Lambda, lam, alpha, B = sp.symbols("Lambda lam alpha B", positive=True)
    v = 2 * r**2 * f * B / lam
    radial_energy = alpha * r**2 / (16 * sp.pi * sp.sqrt(f)) * (
        2 * f * u**2 * Lambda / r**2 + v**2 * Lambda * lam / (2 * r**4))
    zero("normalized axial radial energy", radial_energy
         - Lambda * alpha * sp.sqrt(f) / (8 * sp.pi) * (u**2 + r**2 * f * B**2 / lam))


def main():
    started = time.perf_counter()
    check_conformal_equation()
    check_ground_state_and_mass()
    check_axial_tensors()
    print(json.dumps({"status": "passed", "checks": len(CHECKS),
                      "elapsed_seconds": time.perf_counter() - started,
                      "identities": CHECKS,
                      "scope": "exact algebra only; existence and averaging require analysis"}, indent=2))


if __name__ == "__main__":
    main()
