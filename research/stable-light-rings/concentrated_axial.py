"""Unit-energy radiation layers and their static radial Green responses.

The two radial pieces obey the homogeneous mean equations. The interface
jump is the energy and radial-pressure source of the radial field equations.
At most 2000 paired mesh points are allowed, hence 4000 radial points total.
"""

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq

from boson_star import SINGULAR
from mean_response import background_jacobian, tail_variation
from trapped_axial import gauss_mesh


def bottom_action_variation(background, metric, order=160):
    """Apply the bottom-of-well action variation to a radial metric response."""
    bg, rs = background, background.rs
    outer = brentq(lambda r: bg.fields(r)["U"] - bg.Us, bg.ru, bg.radius)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    t = (nodes + 1) / 2
    r = rs + (outer - rs) * (2 * t - t**2)
    measure = weights * (outer - rs) * (1 - t)
    g = bg.fields(r)
    gap = g["U"] - bg.Us
    near = r - rs < .002
    if np.any(near):
        # Integrating U' avoids cancellation at the double turning point.
        qnodes, qweights = np.polynomial.legendre.leggauss(8)
        length = r[near] - rs
        points = rs + length[:, None] * (qnodes + 1) / 2
        fields = bg.fields(points.ravel())
        Up = 2 * fields["U"] * (fields["nup"] - 1 / points.ravel())
        gap[near] = length / 2 * (Up.reshape(points.shape) @ qweights)
    if np.any(gap <= 0):
        raise ValueError("The limiting barrier has an invalid potential gap.")
    x, v = metric(r)
    vs = metric(rs)[1]
    integrand = (x / (r * g["f"]) * gap + bg.Us * (v - vs)) / (g["p"] * np.sqrt(gap))
    return float(np.dot(measure, integrand)), float(outer)


class ConcentratedResponse:
    """Set fixed_charge=False for the auxiliary response with B(0)=0.

    An explicit source_radius selects a Green datum. Away from a circular
    orbit it need not be a separately conserved physical stress tensor.
    """

    def __init__(self, background, tolerance=1e-7, max_radial_points=4000,
                 fixed_charge=True, light_ring="stable", *, source_radius=None,
                 pressure_fraction=0.):
        if light_ring not in ("stable", "unstable"):
            raise ValueError("Choose the stable or unstable light ring.")
        if not 700 <= max_radial_points <= 4000:
            raise ValueError("The radial point budget must be between 700 and 4000.")
        if not np.isfinite(pressure_fraction) or not 0 <= pressure_fraction <= 1:
            raise ValueError("The radial pressure fraction must lie in [0,1].")
        if source_radius is None and pressure_fraction != 0:
            raise ValueError("A circular null source must have zero radial pressure.")
        if source_radius is not None and not 0 < source_radius < background.radius:
            raise ValueError("The layer must lie strictly inside the radial domain.")
        self.background = background
        self.fixed_charge = fixed_charge
        self.light_ring = light_ring if source_radius is None else "general"
        self.pressure_fraction = pressure_fraction
        self.location = ((background.rs if light_ring == "stable" else background.ru)
                         if source_radius is None else float(source_radius))
        g = background.fields(self.location)
        self.zs = g["alpha"] / np.sqrt(g["f"])
        self.jump = np.array([0, (1-pressure_fraction)*background.sol(self.location)[1]/g["f"],
                              1, -pressure_fraction/g["f"], 0]) / (self.location * self.zs)
        self.widths = (self.location, background.radius - self.location)
        self.singular = np.zeros((5, 5))
        self.singular[:4, :4] = SINGULAR
        singular = np.zeros((10, 10))
        singular[:5, :5] = self.singular

        def bc(left, right):
            constraint = right[9] if fixed_charge else left[0]
            return np.r_[left[1], left[2], left[4], constraint,
                         tail_variation(background, right[5:]),
                         left[5:] - right[:5] - self.jump]

        mesh = np.unique(np.r_[np.linspace(0, 1, 301), .0001 * 2**np.arange(5)])
        self.solution = solve_bvp(self.equation, bc, mesh, np.zeros((10, mesh.size)),
                                  S=singular, tol=tolerance,
                                  max_nodes=max_radial_points // 2)
        if not self.solution.success:
            raise RuntimeError(f"Concentrated response failed within {max_radial_points} radial points: "
                               + self.solution.message)
        end = self.solution.sol(1)[5:]
        self.dlogomega = end[3] - end[2] / background.fields(background.radius)["f"]

    def homogeneous(self, r, y, include_singular=True):
        bg = self.background
        g = bg.fields(r)
        result = np.einsum("ijn,jn->in", background_jacobian(bg, r), y[:4])
        charge = 4 * np.pi * r**2 * g["Omega"] / np.sqrt(g["f"]) * (
            2 * g["A"] * bg.central * y[0] + g["A"]**2 * (y[3] + y[2] / g["f"]))
        result = np.vstack((result, charge))
        if include_singular:
            result += self.singular @ y / r
        return result

    def equation(self, t, y):
        inner = self.widths[0] * self.homogeneous(self.widths[0] * t, y[:5], False)
        # Preserve the exact endpoint despite rounding in location + width.
        outer_r = np.where(t == 1, self.background.radius, self.location + self.widths[1] * t)
        outer = self.widths[1] * self.homogeneous(outer_r, y[5:])
        return np.vstack((inner, outer))

    def state(self, r, derivative=0, side="right"):
        r = np.asarray(r, dtype=float)
        if np.any(r < 0) or np.any(r > self.background.radius):
            raise ValueError("Response evaluation exceeds the radial domain.")
        inner = r < self.location if side == "right" else r <= self.location
        left = self.solution.sol(np.clip(r / self.widths[0], 0, 1), derivative)[:5] / self.widths[0]**derivative
        right = self.solution.sol(np.clip((r - self.location) / self.widths[1], 0, 1), derivative)[5:] / self.widths[1]**derivative
        return np.where(inner, left, right)

    def metric(self, r, side="right"):
        y = self.state(r, side=side)
        return r * y[2], self.dlogomega - y[3]

    def limiting_action(self, order=160):
        """Double-inner-turning-point action derivative for the limiting response.

        This is a functional of the distributional solution. Convergence of
        the physical finite-frequency feedback to it is a separate question.
        """
        if self.light_ring != "stable":
            raise ValueError("The bottom-of-well action requires the stable light ring.")
        return bottom_action_variation(self.background, self.metric, order)

    def diagnostics(self):
        bg, rs = self.background, self.location
        left, right = self.state(rs, side="left"), self.state(rs)
        dl, dr = self.state(rs, 1, "left"), self.state(rs, 1)
        g = bg.fields(rs)
        slope_jump = -dr[3] + dl[3]
        expected_jump = 3 / (rs**2 * g["p"])
        jump_error = np.max(np.abs(right - left - self.jump))
        action, outer = self.limiting_action()
        action_fine, _ = self.limiting_action(240)
        t, weights = gauss_mesh(self.solution.x)
        y, dy = self.solution.sol(t), self.solution.sol(t, 1)
        rhs = self.equation(t, y)
        rhs[:5] += self.singular @ y[:5] / t
        residual = float(np.max(np.abs(dy - rhs)))
        charge = self.widths[0] * np.dot(weights, self.homogeneous(rs * t, y[:5])[4])
        charge += self.widths[1] * np.dot(weights, self.homogeneous(rs + self.widths[1] * t, y[5:])[4])
        interval = np.linspace(rs, outer, 1001)
        x, _ = self.metric(interval)
        return {"radial_points": int(2 * self.solution.x.size),
                "max_rms_residual": float(np.max(self.solution.rms_residuals)),
                "max_scaled_ODE_residual": residual, "interface_error": float(jump_error),
                "added_mass": float(bg.radius * self.state(bg.radius)[2]),
                "charge_integral": float(charge),
                "central_amplitude_response": float(bg.central * self.state(0)[0]),
                "relative_scalar_frequency_response": float(self.dlogomega),
                "mass_response_left": float(rs * left[2]), "mass_response_right": float(rs * right[2]),
                "lapse_derivative_left": float(-dl[3]), "lapse_derivative_right": float(-dr[3]),
                "lapse_derivative_jump": float(slope_jump), "expected_lapse_derivative_jump": float(expected_jump),
                "minimum_exterior_barrier_mass_sampled": float(np.min(x)),
                "minimum_exterior_barrier_lapse_derivative_sampled": float(np.min(-self.state(interval, 1)[3])),
                "limiting_barrier_action_response": action,
                "action_quadrature_change": float(abs(action - action_fine)),
                "outer_turning_point_at_U_s": outer}


if __name__ == "__main__":
    import json
    from trapped_axial import Background
    response = ConcentratedResponse(Background())
    print(json.dumps(response.diagnostics(), indent=2))
