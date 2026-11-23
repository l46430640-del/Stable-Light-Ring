"""Finite-width high-frequency axial source and its monopole response.

The smooth angle coordinate resolves integrable turning-point singularities.
The response uses the represented background and finite endpoint conditions;
its floating-point output does not certify the sign on the exact star.
"""

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq

from finite_static_inverse import WEIGHTS, endpoint_rows
from static_transversality import static_matrix
from trapped_axial import gauss_mesh


class OrbitSource:
    def __init__(self, background, fraction, order=160):
        if not 0 < fraction < 1:
            raise ValueError("The energy must lie strictly inside the trapping well.")
        self.background, self.fraction = background, float(fraction)
        self.energy = background.Us + fraction * (background.Uu - background.Us)
        root = lambda r: float(background.fields(r)["U"] - self.energy)
        self.a = brentq(root, background.sol.x[1], background.rs, xtol=5e-14)
        self.b = brentq(root, background.rs, background.ru, xtol=5e-14)
        self.c = brentq(root, background.ru, background.radius, xtol=5e-14)
        self._nodes, self._weights = np.polynomial.legendre.leggauss(8)
        r, weight, gap = self.raw_quadrature(order)
        self.normalization = float(np.sum(weight))
        self.action = float(np.dot(weight, gap))

    def optical_derivative(self, r):
        # Differentiate the represented potential itself for endpoint cancellation.
        return self.background.fields(r)["U"] * (-2 * self.background.sol(r, 1)[3] - 2 / r)

    def mapped(self, t, barrier=False):
        t = np.atleast_1d(t)
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError("The angle coordinate must be in [0,1].")
        left, right = (self.b, self.c) if barrier else (self.a, self.b)
        sign = 1 if barrier else -1
        width = right - left
        sine, cosine = np.sin(np.pi * t / 2), np.cos(np.pi * t / 2)
        distance = np.where(t <= .5, width * sine**2, -width * cosine**2)
        origin = np.where(t <= .5, left, right)
        r = origin + distance
        dr = np.pi * width * sine * cosine
        gap = sign * (self.background.fields(r)["U"] - self.energy)
        near = np.abs(distance) < .001
        if np.any(near):
            points = origin[near, None] + distance[near, None] * (self._nodes + 1) / 2
            derivative = self.optical_derivative(points.ravel()).reshape(points.shape)
            gap[near] = sign * distance[near] / 2 * (derivative @ self._weights)
        endpoints = (t == 0) | (t == 1)
        gap[endpoints] = 0
        if np.any(gap[~endpoints] <= 0):
            raise ValueError("The mapped potential gap must be positive in the interior.")
        ratio = np.divide(dr, np.sqrt(gap), out=np.zeros_like(dr), where=~endpoints)
        ratio[endpoints] = np.pi * np.sqrt(width / np.abs(self.optical_derivative(origin[endpoints])))
        dr[endpoints] = 0
        return r, dr, gap, ratio

    def raw_quadrature(self, order=160, barrier=False):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        r, _, gap, ratio = self.mapped((nodes + 1) / 2, barrier)
        return r, weights * ratio / (2 * self.background.fields(r)["p"]), gap

    def measure(self, t):
        r, dr, gap, ratio = self.mapped(t)
        density = ratio / (self.background.fields(r)["p"] * self.normalization)
        return r, dr, gap, density

    def forcing(self, t):
        r, dr, gap, density = self.measure(t)
        g = self.background.fields(r)
        z = g["alpha"] / np.sqrt(g["f"])
        pressure = gap / self.energy
        b = self.background.sol(r)[1]
        vector = np.array([np.zeros_like(r), (1 - pressure) * b / g["f"],
                           np.ones_like(r), -pressure / g["f"]]) / (r * z)
        return r, dr, vector * density


class FiniteWidthResponse:
    def __init__(self, source, tolerance=2e-6, max_radial_points=4000):
        if not 0 < tolerance < 1 or not 1200 <= max_radial_points <= 4000:
            raise ValueError("Invalid tolerance or radial point budget.")
        self.source, self.background = source, source.background
        self.inner_radius = .002
        if not self.inner_radius < source.a < source.b < source.c <= self.background.radius:
            raise ValueError("The three turning points must lie inside the response domain.")
        self.tolerance = tolerance
        self.widths = (source.a - self.inner_radius, self.background.radius - source.b)
        self.left_rows, self.right_rows, self.right_rhs = endpoint_rows(self.background, self.inner_radius)
        mesh = np.unique(np.r_[np.linspace(0, 1, 351), np.geomspace(1e-6, .01, 25)])
        self.solution = solve_bvp(self.equation, self.boundary, mesh, np.zeros((12, mesh.size)),
                                  fun_jac=self.jacobian, tol=tolerance,
                                  max_nodes=max_radial_points // 3)
        if not self.solution.success:
            raise RuntimeError(f"Finite-width response stopped within {max_radial_points} radial points: "
                               + self.solution.message)
        end = WEIGHTS * self.solution.sol(1)[8:]
        self.dlogomega = end[3] - end[2] / self.background.fields(self.background.radius)["f"]

    def geometry(self, t):
        middle, middle_dr, forcing = self.source.forcing(t)
        # Set the exact endpoints before evaluating a strictly bounded background.
        inner = np.where(t == 1, self.source.a, self.inner_radius + self.widths[0] * t)
        outer = np.where(t == 1, self.background.radius, self.source.b + self.widths[1] * t)
        radii = (inner, middle, outer)
        slopes = (self.widths[0], middle_dr, self.widths[1])
        return radii, slopes, forcing

    def equation(self, t, y):
        radii, slopes, forcing = self.geometry(t)
        rhs = np.empty_like(y)
        for i, (r, slope) in enumerate(zip(radii, slopes)):
            state = WEIGHTS[:, None] * y[4*i:4*i+4]
            matrix = static_matrix(self.background, r)[:, :4, :4]
            value = slope * np.einsum("nij,jn->in", matrix, state)
            if i == 1:
                value += forcing
            rhs[4*i:4*i+4] = value / WEIGHTS[:, None]
        return rhs

    def jacobian(self, t, y):
        radii, slopes, _ = self.geometry(t)
        result = np.zeros((12, 12, t.size))
        for i, (r, slope) in enumerate(zip(radii, slopes)):
            matrix = np.moveaxis(static_matrix(self.background, r)[:, :4, :4], 0, -1)
            result[4*i:4*i+4, 4*i:4*i+4] = matrix * slope * WEIGHTS[None, :, None] / WEIGHTS[:, None, None]
        return result

    def boundary(self, left, right):
        return np.r_[self.left_rows @ left[:4], self.right_rows @ right[8:] - self.right_rhs,
                     left[4:8] - right[:4], left[8:] - right[4:8]]

    def state(self, r):
        r = np.atleast_1d(r)
        if np.any(r < self.inner_radius) or np.any(r > self.background.radius):
            raise ValueError("Response evaluation exceeds the finite radial domain.")
        result = np.empty((4, r.size))
        masks = (r < self.source.a, (r >= self.source.a) & (r <= self.source.b), r > self.source.b)
        for i, mask in enumerate(masks):
            if not np.any(mask):
                continue
            if i == 0:
                t = (r[mask] - self.inner_radius) / self.widths[0]
            elif i == 1:
                t = 2 / np.pi * np.arcsin(np.sqrt(np.clip((r[mask] - self.source.a) /
                                                         (self.source.b - self.source.a), 0, 1)))
            else:
                t = (r[mask] - self.source.b) / self.widths[1]
            result[:, mask] = WEIGHTS[:, None] * self.solution.sol(t)[4*i:4*i+4]
        return result

    def metric(self, r):
        state = self.state(r)
        x, v = np.asarray(r) * state[2], self.dlogomega - state[3]
        if np.ndim(r) == 0:
            return float(x[0]), float(v[0])
        return x, v

    def action_response(self, order=160):
        source, bg = self.source, self.background
        r, weights, gap = source.raw_quadrature(order)
        weights = weights / np.sum(weights)
        state = self.state(r)
        k, v = state[2] / bg.fields(r)["f"], self.dlogomega - state[3]
        vstar = float(np.dot(weights, v - k * gap / source.energy))
        r, weights, gap = source.raw_quadrature(order, barrier=True)
        state = self.state(r)
        k, v = state[2] / bg.fields(r)["f"], self.dlogomega - state[3]
        mass_term = float(np.dot(weights, k * gap))
        lapse_term = float(np.dot(weights, source.energy * (v - vstar)))
        return mass_term + lapse_term, mass_term, lapse_term, vstar

    def diagnostics(self):
        source, bg = self.source, self.background
        t, weights = gauss_mesh(self.solution.x, order=4)
        y, derivative = self.solution.sol(t), self.solution.sol(t, 1)
        rhs = self.equation(t, y)
        defect = np.abs(derivative-rhs)
        residual = float(np.max(defect))
        relative_residual = float(np.max(defect/(1+np.abs(rhs))))
        radii, slopes, _ = self.geometry(t)
        charge = 0.
        for i, (r, slope) in enumerate(zip(radii, slopes)):
            charge_rate = static_matrix(bg, r)[:, 4, :4]
            state = WEIGHTS[:, None] * y[4*i:4*i+4]
            charge += np.dot(weights, slope * np.einsum("ni,in->n", charge_rate, state))
        # The regular central Taylor charge is retained below the finite endpoint.
        central = self.state(self.inner_radius)[:, 0]
        g0 = bg.fields(0.)
        charge += 4 * np.pi / 3 * g0["Omega"] * bg.central**2 * self.inner_radius**3 * (2 * central[0] + central[3])
        action, mass, lapse, vstar = self.action_response()
        finer_action = self.action_response(240)[0]
        r, weight, gap = source.raw_quadrature(240)
        pressure = float(np.dot(weight, gap) / (source.energy * np.sum(weight)))
        sample = np.linspace(source.b, source.c, 401)
        state = self.state(sample)
        slope = np.einsum("nij,jn->in", static_matrix(bg, sample)[:, :4, :4], state)
        return {"energy_fraction": source.fraction, "energy": source.energy,
                "turning_points": [source.a, source.b, source.c],
                "normalization_Z": source.normalization, "well_action": source.action,
                "radial_pressure_fraction": pressure,
                "normalization_quadrature_change": float(abs(np.sum(weight) / source.normalization - 1)),
                "pressure_identity_error": float(abs(pressure - source.action / (source.energy * source.normalization))),
                "tolerance": self.tolerance, "radial_points": int(3 * self.solution.x.size),
                "max_rms_residual": float(np.max(self.solution.rms_residuals)),
                "max_angle_ODE_residual": residual,
                "max_relative_angle_ODE_residual": relative_residual,
                "boundary_matching_residual": float(np.max(np.abs(self.boundary(self.solution.sol(0), self.solution.sol(1))))),
                "added_mass": float(bg.radius * self.state(bg.radius)[2, 0]),
                "charge_integral": float(charge),
                "barrier_mass_ratio_min_sampled": float(np.min(state[2])),
                "barrier_lapse_slope_min_sampled": float(np.min(-slope[3])),
                "barrier_lapse_minus_vstar_min_sampled": float(np.min(self.dlogomega - state[3] - vstar)),
                "delta_barrier_action": action, "mass_action_term": mass, "lapse_action_term": lapse,
                "action_quadrature_change": float(abs(action - finer_action))}
