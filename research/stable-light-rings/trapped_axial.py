"""Localized axial modes on the verified boson star; no outgoing resonance claim.

The finite-interval eigenfunction is smoothly cut off inside the forbidden
regions. Both the cutoff defect and the collocation defect are measured.
"""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import PPoly
from scipy.optimize import brentq

from boson_star import potential


class Background:
    """Saved S08A06 background, or a supplied solution with light-ring brackets."""
    def __init__(self, path=None, solution=None, central=0.06, scale=0.08, ring_brackets=None):
        if solution is None:
            path = path or Path(__file__).parent / "data/s08a06/background.npz"
            with np.load(path) as data:
                self.sol = PPoly.construct_fast(data["coefficients"].copy(),
                                               data["r"].copy(), False, axis=1)
        else:
            self.sol = solution.sol
        self.central, self.scale = central, scale
        self.radius = float(self.sol.x[-1])
        y = self.sol(self.radius)
        self.omega = np.exp(y[3]) * np.sqrt(1 - 2 * y[2])
        self.mass = self.radius * y[2]
        brackets = ((15.8, 16.1), (16.4, 16.7)) if ring_brackets is None else ring_brackets
        if len(brackets) != 2 or not 0 < brackets[0][0] < brackets[0][1] <= brackets[1][0] < brackets[1][1] <= self.radius:
            raise ValueError("Two ordered light-ring brackets inside the radial domain are required.")
        self.rs = brentq(lambda r: self.fields(r)["nup"] - 1 / r, *brackets[0])
        self.ru = brentq(lambda r: self.fields(r)["nup"] - 1 / r, *brackets[1])
        self.Us = self.fields(self.rs)["U"]
        self.Uu = self.fields(self.ru)["U"]

    def fields(self, r):
        r = np.asarray(r)
        if np.any(r < 0) or np.any(r > self.radius):
            raise ValueError("Background evaluation exceeds its verified radial domain.")
        a, b, u, zeta = self.sol(r)
        A, Ap = self.central * a, self.central * b
        f, Omega = 1 - 2 * u, np.exp(zeta)
        alpha = self.omega / Omega
        V, Vs = potential(A, self.scale)
        W, G = Omega**2 * A**2, f * Ap**2
        rho, pr = (W + G + V) / 2, (W + G - V) / 2
        invr = np.divide(1., r, out=np.zeros_like(r, dtype=float), where=r != 0)
        nup = (u * invr + 4 * np.pi * r * pr) / f
        fp = -8 * np.pi * r * rho + 2 * u * invr
        p = alpha * np.sqrt(f)
        return {"A": A, "Ap": Ap, "u": u, "f": f, "Omega": Omega,
                "alpha": alpha, "rho": rho, "pr": pr, "V": V, "Vs": Vs,
                "nup": nup, "fp": fp, "p": p, "dlogp": nup + fp / (2 * f),
                "U": alpha**2 * invr**2}

    def axial_potential(self, r, ell):
        g = self.fields(r)
        return g["alpha"]**2 * (ell * (ell + 1) / r**2
                                - 6 * g["u"] / r**2 + 4 * np.pi * g["V"])


def cutoff(r, left=(15.1, 15.3), right=(16.5, 16.7)):
    """C2 cutoff and its first two radial derivatives."""
    r = np.asarray(r)
    value, first, second = np.ones_like(r), np.zeros_like(r), np.zeros_like(r)
    for start, end, increasing in ((*left, True), (*right, False)):
        width = end - start
        t = np.clip((r - start) / width, 0, 1)
        s = 10 * t**3 - 15 * t**4 + 6 * t**5
        sp = 30 * t**2 * (1 - t)**2 / width
        spp = 60 * t * (1 - t) * (1 - 2 * t) / width**2
        if not increasing:
            s, sp, spp = 1 - s, -sp, -spp
        second, first, value = (second * s + 2 * first * sp + value * spp,
                                first * s + value * sp, value * s)
    return value, first, second


def gauss_mesh(knots, order=8):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    mid, half = (knots[:-1] + knots[1:]) / 2, np.diff(knots) / 2
    return ((mid[:, None] + half[:, None] * nodes).ravel(),
            (half[:, None] * weights).ravel())


class TrappedAxial:
    def __init__(self, background, ell=50000, tolerance=1e-7,
                 interval=(15.0, 16.75), max_nodes=4000):
        if ell < 2 or ell != int(ell):
            raise ValueError("An axial radiative harmonic requires an integer ell >= 2.")
        if interval[0] >= 15.1 or interval[1] <= 16.7 or interval[0] <= 0:
            raise ValueError("The eigenfunction interval must contain the complete cutoff support.")
        self.background, self.ell = background, ell
        L = ell + 0.5
        g = background.fields(background.rs)
        Fprime = 8 * np.pi * background.rs**2 * g["Omega"]**2 * g["A"]**2 - 1
        Upp = 2 * background.Us * Fprime / (background.rs**2 * g["f"])
        gamma = L * np.sqrt(Upp / 2) / g["p"]
        r = np.linspace(*interval, 601)
        psi = np.exp(-gamma * (r - background.rs)**2 / 2)
        psi *= np.sqrt(g["p"]) * (gamma / np.pi)**0.25
        dpsi = -gamma * (r - background.rs) * psi
        from scipy.special import erf
        integral = (1 + erf(np.sqrt(gamma) * (r - background.rs))) / 2
        delta = L * g["p"] * np.sqrt(Upp / 2)
        self.base_frequency2 = L**2 * background.Us

        def equation(r, y, parameters):
            g = background.fields(r)
            excess = background.axial_potential(r, ell) - self.base_frequency2
            return np.array([y[1], -g["dlogp"] * y[1]
                             + (excess - parameters[0]) * y[0] / g["p"]**2,
                             y[0]**2 / g["p"]])

        self.solution = solve_bvp(equation, lambda a, b, p: np.array([a[0], b[0], a[2], b[2] - 1]),
                                  r, np.array([psi, dpsi, integral]), p=[delta],
                                  tol=tolerance, max_nodes=max_nodes)
        if not self.solution.success:
            raise RuntimeError(f"Axial BVP failed within {max_nodes} nodes: {self.solution.message}")
        self.sigma2 = self.base_frequency2 + self.solution.p[0]
        self.sigma = np.sqrt(self.sigma2)
        self.factor = 1.0
        self.knots = np.unique(np.r_[self.solution.x, 15.1, 15.3, 16.5, 16.7])
        x, weights = gauss_mesh(self.knots)
        source = self.source(x)
        g = background.fields(x)
        energy = np.dot(weights, 4 * np.pi * x**2 * g["alpha"] / np.sqrt(g["f"]) * source[0])
        self.factor = 1 / np.sqrt(energy)

    def profile(self, r):
        r = np.asarray(r, dtype=float)
        # The cutoff is zero before either finite-interval boundary is reached.
        interior = np.clip(r, self.solution.x[0], self.solution.x[-1])
        psi, dpsi, _ = self.solution.sol(interior)
        c, cp, cpp = cutoff(r)
        return self.factor * c * psi, self.factor * (c * dpsi + cp * psi)

    def source(self, r):
        r = np.asarray(r, dtype=float)
        g = self.background.fields(r)
        psi, dpsi = self.profile(r)
        invr = np.divide(1., r, out=np.zeros_like(r), where=r != 0)
        H1 = r * psi / g["p"]
        H0 = g["p"] * (psi + r * dpsi) / self.sigma
        lam = self.ell * (self.ell + 1) - 2
        # The curvature amplitude is evaluated from its definition, including
        # the cutoff defect, rather than imposing the axial equation off shell.
        interior = np.clip(r, self.solution.x[0], self.solution.x[-1])
        raw, rawp, _ = self.solution.sol(interior)
        gi = self.background.fields(interior)
        rawpp = (-gi["dlogp"] * rawp
                 + (self.background.axial_potential(interior, self.ell) - self.sigma2)
                 * raw / gi["p"]**2)
        c, cp, cpp = cutoff(r)
        ddpsi = self.factor * (c * rawpp + 2 * cp * rawp + cpp * raw)
        H0p = g["p"] / self.sigma * (g["dlogp"] * (psi + r * dpsi) + 2 * dpsi + r * ddpsi)
        Xi = -self.sigma * H1 - H0p + 2 * H0 * invr
        electric = g["f"] / g["alpha"]**2 * Xi**2
        temporal = lam * invr**2 * H0**2 / g["alpha"]**2
        radial = lam * invr**2 * g["f"] * H1**2
        prefactor = invr**2 / (64 * np.pi)  # c_ell=1 and period average 1/2.
        rho = prefactor * (electric + temporal + radial)
        pr = prefactor * (-electric + temporal + radial)
        pt = prefactor * (2 * electric + temporal - radial)
        xc = r * g["f"]**2 * H1**2 * invr**2 / 4
        vc = -H0**2 * invr**2 / (4 * g["alpha"]**2)
        return rho, pr, pt, xc, vc

    def diagnostics(self):
        r, weights = gauss_mesh(self.knots)
        g = self.background.fields(r)
        psi, psip, _ = self.solution.sol(r)
        derivative = self.solution.sol(r, 1)
        background_derivative = self.background.sol(r, 1)
        polynomial_dlogp = -background_derivative[3] - background_derivative[2] / g["f"]
        c, cp, cpp = cutoff(r)
        residual = (-g["p"]**2 * derivative[1] - g["p"]**2 * g["dlogp"] * psip
                    + (self.background.axial_potential(r, self.ell) - self.sigma2) * psi)
        # Use the represented polynomial's own first and second derivatives
        # as a separate operator residual, independent of the auxiliary state.
        polynomial_residual = (-g["p"]**2 * self.solution.sol(r, 2)[0]
                               - g["p"]**2 * polynomial_dlogp * derivative[0]
                               + (self.background.axial_potential(r, self.ell) - self.sigma2) * psi)
        cut_residual = -g["p"]**2 * (cpp * psi + 2 * cp * psip + g["dlogp"] * cp * psi)
        full_residual = c * polynomial_residual - g["p"]**2 * (
            cpp * psi + 2 * cp * derivative[0] + polynomial_dlogp * cp * psi)
        norm = lambda a: np.sqrt(np.dot(weights, a**2 / g["p"]))
        norm_cut = norm(c * psi)
        E = self.sigma2 / (self.ell + .5)**2
        rho, _, _, _, _ = self.source(r)
        return {"ell": self.ell, "sigma": float(self.sigma), "reduced_energy": float(E),
                "fraction_of_well_depth": float((E - self.background.Us) / (self.background.Uu - self.background.Us)),
                "nodes": int(self.solution.x.size),
                "max_rms_residual": float(np.max(self.solution.rms_residuals)),
                "cutoff_operator_defect": float(norm(cut_residual) / norm_cut),
                "collocation_operator_defect": float(norm(c * residual) / norm_cut),
                "polynomial_operator_defect": float(norm(c * polynomial_residual) / norm_cut),
                "full_cut_polynomial_operator_defect": float(norm(full_residual) / norm_cut),
                "linear_energy_error_rate": float(norm(full_residual) / (self.sigma * norm_cut)),
                "profile_first_derivative_defect": float(norm(c * (derivative[0] - psip)) / norm_cut),
                "energy_integral": float(np.dot(weights, 4 * np.pi * r**2 * g["alpha"] / np.sqrt(g["f"]) * rho)),
                "mode_nodes_in_well": int(np.sum(np.diff(np.sign(self.solution.sol(np.linspace(15.5, 16.4, 501))[0])) != 0))}


def action_data(background, ell, v=None, k=None, breakpoints=()):
    """Leading WKB ground action and, optionally, its fixed-action response."""
    L = ell + .5

    def points(E):
        fun = lambda r: background.fields(r)["U"] - E
        return (brentq(fun, 15, background.rs), brentq(fun, background.rs, background.ru),
                brentq(fun, background.ru, 17.2))

    def integrate(function, lo, hi):
        # Gaussian quadrature avoids evaluating the removable endpoint factors.
        t, weights = np.polynomial.legendre.leggauss(160)
        theta = (t + 1) * np.pi / 4
        cuts = [lo] + sorted(point for point in breakpoints if lo < point < hi) + [hi]
        value = 0.0
        for left, right in zip(cuts[:-1], cuts[1:]):
            r = left + (right - left) * np.sin(theta)**2
            measure = (right - left) * np.sin(2 * theta) * weights * np.pi / 4
            value += np.dot(measure, function(r))
        return value

    def J(E):
        a, b, _ = points(E)
        return integrate(lambda r: np.sqrt(np.maximum(E - background.fields(r)["U"], 0))
                         / background.fields(r)["p"], a, b)

    E = brentq(lambda energy: L * J(energy) - np.pi / 2,
               background.Us + (background.Uu - background.Us) * 1e-4,
               background.Uu - (background.Uu - background.Us) * 1e-4, xtol=1e-16)
    a, b, c = points(E)
    S = integrate(lambda r: np.sqrt(np.maximum(background.fields(r)["U"] - E, 0))
                  / background.fields(r)["p"], b, c)
    result = {"energy": float(E), "turning_points": [float(a), float(b), float(c)],
              "well_action": float(J(E)), "barrier_action": float(S), "barrier_exponent": float(2 * L * S)}
    if v is not None and k is not None:
        weight = lambda r: 1 / (background.fields(r)["p"] * np.sqrt(E - background.fields(r)["U"]))
        vstar = integrate(lambda r: weight(r) * (v(r) - k(r) * (1 - background.fields(r)["U"] / E)), a, b)
        vstar /= integrate(weight, a, b)
        dS = integrate(lambda r: (k(r) * (background.fields(r)["U"] - E) + E * (v(r) - vstar))
                       / (background.fields(r)["p"] * np.sqrt(background.fields(r)["U"] - E)), b, c)
        result.update({"vstar": float(vstar), "delta_barrier_action": float(dS)})
    return result


if __name__ == "__main__":
    import json
    bg = Background()
    for ell in (50000, 100000):
        mode = TrappedAxial(bg, ell)
        print(json.dumps({"mode": mode.diagnostics(), "WKB": action_data(bg, ell)}, indent=2))
