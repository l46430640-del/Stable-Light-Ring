"""Bounded radial collocation for the solitonic Einstein--scalar background.

Units have mu=1. The unknowns are A/A0, (A/A0)', m/r, and log(omega/alpha).
The finite-radius exterior condition must be checked by changing the domain.
"""

import argparse
import json
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad, simpson, solve_bvp
from scipy.interpolate import PPoly
from scipy.optimize import brentq


def potential(A, scale):
    s = A**2
    V = s * (1 - 2 * s / scale**2)**2
    Vs = (1 - 2 * s / scale**2) * (1 - 6 * s / scale**2)
    return V, Vs


def equations(r, y, central, scale):
    a, b, u, log_local_frequency = y
    A, Ap = central * a, central * b
    f = 1 - 2 * u
    Omega2 = np.exp(2 * log_local_frequency)
    V, Vs = potential(A, scale)
    W, G = Omega2 * A**2, f * Ap**2
    rho, pr = (W + G + V) / 2, (W + G - V) / 2
    invr = np.divide(1.0, r, out=np.zeros_like(r), where=r != 0)
    return np.array([
        b,
        (Vs - Omega2) * a / f - (2 * u * invr + 4 * np.pi * r * (pr - rho)) * b / f,
        4 * np.pi * r * rho,
        -(2 * u**2 * invr + 4 * np.pi * r * pr) / f,
    ])


SINGULAR = np.zeros((4, 4))
SINGULAR[1, 1] = -2
SINGULAR[2, 2] = -1
SINGULAR[3, 2] = -1


def boundary(left, right, radius):
    a, b, u, log_Omega = right
    omega2 = np.exp(2 * log_Omega) * (1 - 2 * u)
    # Continuation of the real boundary residual outside the physical branch.
    # Accepted solutions are separately required to have 0 < omega^2 < 1.
    kappa = np.sqrt(max(1 - omega2, 1e-8))
    M = radius * u
    beta = M * (2 * omega2 - 1) / kappa
    return np.array([left[0] - 1, left[1], left[2], b + (kappa + (1 - beta) / radius) * a])


def initial_guess(r, central, scale, wall):
    Omega = 0.23 + 0.32 * np.exp(-0.5 * (r / 10)**2)
    core = scale * np.sqrt((2 + np.sqrt(1 + 3 * Omega**2)) / 6)
    wall_profile = 1 / np.sqrt(1 + np.exp(2 * (r - wall)))
    a = core * wall_profile
    a /= a[0]
    b = np.gradient(a, r, edge_order=2)
    b[0] = 0
    A, Ap = central * a, central * b
    V, _ = potential(A, scale)
    rho = (Omega**2 * A**2 + Ap**2 + V) / 2
    mass = cumulative_trapezoid(4 * np.pi * r**2 * rho, r, initial=0)
    # This mass only initializes Newton iteration; it is not a boundary condition.
    mass *= 5.65 / mass[-1]
    u = np.divide(mass, r, out=np.zeros_like(r), where=r != 0)
    return np.array([a, b, u, np.log(Omega)])


def solve_background(central=0.06, scale=0.08, radius=40.0, wall=16.0,
                     tolerance=1e-6, max_nodes=4000, previous=None):
    if radius <= 1 or central <= 0 or scale <= 0 or tolerance <= 0:
        raise ValueError("The solver requires a positive field/scale/tolerance and outer radius > 1.")
    r = np.unique(np.r_[np.linspace(0, radius, 401), 0.002 * 2**np.arange(6)])
    guess = initial_guess(r, central, scale, wall) if previous is None else previous.sol(r)
    solution = solve_bvp(lambda x, y: equations(x, y, central, scale),
                         lambda left, right: boundary(left, right, radius),
                         r, guess, S=SINGULAR, tol=tolerance,
                         max_nodes=max_nodes, verbose=0)
    return solution


def diagnostics(solution, central, scale, radius):
    r = np.linspace(0, radius, 8001)
    a, b, u, log_Omega = solution.sol(r)
    A, Ap = central * a, central * b
    f = 1 - 2 * u
    M = radius * u[-1]
    omega2 = np.exp(2 * log_Omega[-1]) * f[-1]
    if np.min(f) <= 0 or not 0 < omega2 < 1:
        raise ValueError("The converged branch is outside the horizonless, decaying scalar domain.")
    omega = np.sqrt(omega2)
    alpha = omega * np.exp(-log_Omega)
    V, _ = potential(A, scale)
    W, G = np.exp(2 * log_Omega) * A**2, f * Ap**2
    rho, pr, pt = (W + G + V) / 2, (W + G - V) / 2, (W - G - V) / 2
    N = simpson(4 * np.pi * r**2 * np.exp(log_Omega) * A**2 / np.sqrt(f), x=r)
    F = r * (3 * u + 4 * np.pi * r**2 * pr - 1)

    def light_ring_function(x):
        a, b, u, log_Omega = solution.sol(x)
        f = 1 - 2 * u
        V, _ = potential(central * a, scale)
        pr = (np.exp(2 * log_Omega) * (central * a)**2 + f * (central * b)**2 - V) / 2
        return 3 * u + 4 * np.pi * x**2 * pr - 1

    rings = []
    for index in np.flatnonzero(F[1:-1] * F[2:] < 0) + 1:
        location = brentq(light_ring_function, r[index], r[index + 1], xtol=1e-12)
        a_lr, _, _, log_lr = solution.sol(location)
        W_lr = np.exp(2 * log_lr) * (central * a_lr)**2
        Fprime = 8 * np.pi * location**2 * W_lr - 1
        rings.append({"r": float(location), "Fprime": float(Fprime), "stable": bool(Fprime > 0)})
    r99 = brentq(lambda x: x * solution.sol(x)[2] - 0.99 * M, 0, radius)
    u_spline = PPoly(solution.sol.c[:, :, 2], solution.x, extrapolate=False)
    critical_u = u_spline.derivative().roots(extrapolate=False)
    critical_u = critical_u[np.isfinite(critical_u)]
    min_f = np.min(1 - 2 * u_spline(np.r_[solution.x, critical_u]))
    a_spline = PPoly(solution.sol.c[:, :, 0], solution.x, extrapolate=False)
    critical_a = a_spline.derivative().roots(extrapolate=False)
    critical_a = critical_a[np.isfinite(critical_a)]
    min_A = central * np.min(a_spline(np.r_[solution.x, critical_a]))
    return {
        "central_amplitude": central, "potential_scale": scale, "outer_radius": radius,
        "success": bool(solution.success), "status": int(solution.status),
        "message": solution.message, "nodes": int(solution.x.size),
        "max_rms_residual": float(np.max(solution.rms_residuals)),
        "mass": float(M), "charge": float(N), "omega": float(omega),
        "central_lapse": float(alpha[0]), "minimum_f": float(min_f),
        "r99": float(r99), "A_outer": float(A[-1]),
        "minimum_A": float(min_A), "light_rings": rings,
    }


def integral_diagnostics(solution, central, scale, omega, order=8):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    mid = (solution.x[1:] + solution.x[:-1]) / 2
    half_width = np.diff(solution.x) / 2
    r = (mid[:, None] + half_width[:, None] * nodes).ravel()
    measure = (half_width[:, None] * weights).ravel()
    a, b, u, log_Omega = solution.sol(r)
    A, Ap, f = central * a, central * b, 1 - 2 * u
    Omega = np.exp(log_Omega)
    alpha = omega / Omega
    V, _ = potential(A, scale)
    W, G = Omega**2 * A**2, f * Ap**2
    rho = (W + G + V) / 2
    mass = np.dot(measure, 4 * np.pi * r**2 * rho)
    charge = np.dot(measure, 4 * np.pi * r**2 * Omega * A**2 / np.sqrt(f))
    komar = np.dot(measure, 4 * np.pi * r**2 * alpha * (2 * W - V) / np.sqrt(f))
    return {"mass_integral": float(mass), "charge_integral": float(charge), "komar_mass": float(komar)}


def residual_diagnostics(solution, central, scale):
    # Points not used as collocation nodes, and an independent angular equation.
    widths = np.diff(solution.x)
    r = np.concatenate([solution.x[:-1] + fraction * widths for fraction in (0.37, 0.79)])
    y, dy, ddy = solution.sol(r), solution.sol(r, 1), solution.sol(r, 2)
    a, b, u, log_Omega = y
    A, Ap, f = central * a, central * b, 1 - 2 * u
    V, Vs = potential(A, scale)
    W, G = np.exp(2 * log_Omega) * A**2, f * Ap**2
    rho, pr, pt = (W + G + V) / 2, (W + G - V) / 2, (W - G - V) / 2
    nup, nupp, fp = -dy[3], -ddy[3], -2 * dy[2]
    mass_error = u + r * dy[2] - 4 * np.pi * r**2 * rho
    lapse_error = nup - (u + 4 * np.pi * r**2 * pr) / (r * f)
    kg_error = f * central * dy[1] + (2 * f / r + f * nup + fp / 2) * Ap + (np.exp(2 * log_Omega) - Vs) * A
    angular_error = (f * (nupp + nup**2 + nup / r) + fp * (nup + 1 / r) / 2) / (8 * np.pi) - pt
    result = {"A_prime_error": float(np.max(np.abs(central * dy[0] - Ap))),
              "mass_equation_error": float(np.max(np.abs(mass_error))),
              "lapse_equation_error": float(np.max(np.abs(lapse_error))),
              "KG_equation_error": float(np.max(np.abs(kg_error))),
              "angular_Einstein_error": float(np.max(np.abs(angular_error)))}
    log0 = solution.sol(0)[3]
    V0, Vs0 = potential(central, scale)
    W0 = np.exp(2 * log0) * central**2
    rho0, pr0 = (W0 + V0) / 2, (W0 - V0) / 2
    center = solution.sol(0, 2)
    result["center_A_second_error"] = float(abs(central * center[0] - (Vs0 - np.exp(2 * log0)) * central / 3))
    result["center_u_second_error"] = float(abs(center[2] - 8 * np.pi * rho0 / 3))
    result["center_log_frequency_second_error"] = float(abs(center[3] + 4 * np.pi * (rho0 / 3 + pr0)))
    return result


def trapping_diagnostics(solution, report):
    rings = report["light_rings"]
    if len(rings) != 2 or not rings[0]["stable"] or rings[1]["stable"]:
        raise ValueError("The selected diagnostic requires one stable/unstable light-ring pair.")
    rs, ru, omega = rings[0]["r"], rings[1]["r"], report["omega"]

    def U(r):
        return omega**2 * np.exp(-2 * solution.sol(r)[3]) / r**2

    Us, Uu = U(rs), U(ru)
    a_top = brentq(lambda r: U(r) - Uu, rs / 2, rs, xtol=1e-12)

    def integrand(theta):
        r = a_top + (ru - a_top) * np.sin(theta)**2
        _, _, u, log_Omega = solution.sol(r)
        q = np.exp(log_Omega) / (omega * np.sqrt(1 - 2 * u))
        dr = 2 * (ru - a_top) * np.sin(theta) * np.cos(theta)
        return q * np.sqrt(max(Uu - U(r), 0.0)) * dr

    action, error = quad(integrand, 0, np.pi / 2, epsabs=1e-13, epsrel=1e-9, limit=100)
    geometric_rings = [brentq(lambda r: -solution.sol(r, 1)[3] - 1 / r,
                             location - 0.1, location + 0.1, xtol=1e-12) for location in (rs, ru)]
    return {"U_stable": float(Us), "U_unstable": float(Uu),
            "relative_well_depth": float((Uu - Us) / Us),
            "inner_turning_point_at_barrier_top": float(a_top),
            "radial_action_at_barrier_top": float(action), "quadrature_error_estimate": float(error),
            "leading_WKB_ground_threshold_L": float(np.pi / (2 * action)),
            "light_ring_derivative_crosscheck": float(np.max(np.abs(np.array(geometric_rings) - [rs, ru])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=40)
    parser.add_argument("--wall", type=float, default=16)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-nodes", type=int, default=4000)
    args = parser.parse_args()
    solution = solve_background(radius=args.radius, wall=args.wall,
                                tolerance=args.tolerance, max_nodes=args.max_nodes)
    if not solution.success:
        print(json.dumps({"success": False, "status": int(solution.status),
                          "message": solution.message, "nodes": int(solution.x.size)}, indent=2))
        raise SystemExit(1)
    print(json.dumps(diagnostics(solution, 0.06, 0.08, args.radius), indent=2))


if __name__ == "__main__":
    main()
