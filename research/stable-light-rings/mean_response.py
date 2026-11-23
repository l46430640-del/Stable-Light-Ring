"""Fixed-total-charge stationary monopole response to a localized axial seed.

This solves the projected, time-averaged quadratic equations. It does not
evolve the nonspherical spacetime or control the averaging error.
"""

import numpy as np
from scipy.integrate import solve_bvp

from boson_star import SINGULAR, equations
from trapped_axial import action_data, gauss_mesh


def background_jacobian(background, r):
    y = background.sol(r).astype(complex)
    columns = []
    for column in range(4):
        shifted = y.copy()
        shifted[column] += 1e-25j
        columns.append(equations(r, shifted, background.central, background.scale).imag / 1e-25)
    return np.stack(columns, axis=1)


def tail_variation(background, variation):
    """Derivative of the same finite-radius scalar condition as the background."""
    da, db, du, dz = variation[:4]
    a, _, u, _ = background.sol(background.radius)
    f, omega, M, R = 1 - 2 * u, background.omega, background.mass, background.radius
    kappa = np.sqrt(1 - omega**2)
    beta = M * (2 * omega**2 - 1) / kappa
    dlogomega = dz - du / f
    dkappa = -omega**2 * dlogomega / kappa
    dbeta = (2 * omega**2 - 1) * R * du / kappa
    dbeta += M * (4 * omega**2 / kappa + (2 * omega**2 - 1) * omega**2 / kappa**3) * dlogomega
    return db + (kappa + (1 - beta) / R) * da + (dkappa - dbeta / R) * a


class MeanResponse:
    def __init__(self, background, mode, tolerance=1e-7, max_nodes=4000):
        self.background, self.mode = background, mode
        r = np.unique(np.round(np.r_[np.linspace(0, background.radius, 401),
                                     np.linspace(15.1, 16.7, 401), .002 * 2**np.arange(6)], 12))
        singular = np.zeros((5, 5))
        singular[:4, :4] = SINGULAR
        self.singular = singular

        def bc(left, right):
            return np.array([left[1], left[2], left[4], right[4], tail_variation(background, right)])

        self.solution = solve_bvp(self.equation, bc, r, np.zeros((5, r.size)),
                                  S=singular, tol=tolerance, max_nodes=max_nodes)
        if not self.solution.success:
            raise RuntimeError(f"Mean response failed within {max_nodes} nodes: {self.solution.message}")
        end = self.solution.sol(background.radius)
        self.dlogomega = end[3] - end[2] / background.fields(background.radius)["f"]

    def equation(self, r, y):
        bg = self.background
        g = bg.fields(r)
        rho, pr, _, _, _ = self.mode.source(r)
        rhs = np.einsum("ijn,jn->in", background_jacobian(bg, r), y[:4])
        rhs[1] -= 4 * np.pi * r * (pr - rho) * bg.sol(r)[1] / g["f"]
        rhs[2] += 4 * np.pi * r * rho
        rhs[3] -= 4 * np.pi * r * pr / g["f"]
        charge = 4 * np.pi * r**2 * g["Omega"] / np.sqrt(g["f"]) * (
            2 * g["A"] * bg.central * y[0] + g["A"]**2 * (y[3] + y[2] / g["f"]))
        return np.vstack((rhs, charge))

    def metric(self, r, completed=False):
        y = self.solution.sol(r)
        x, v = r * y[2], self.dlogomega - y[3]
        if not completed:
            _, _, _, xc, vc = self.mode.source(r)
            x, v = x + xc, v + vc
        return x, v

    def diagnostics(self):
        bg = self.background
        knots = np.unique(np.r_[self.solution.x, self.mode.knots])
        r, weights = gauss_mesh(knots)
        y, dy = self.solution.sol(r), self.solution.sol(r, 1)
        g = bg.fields(r)
        rho, pr, pt, _, _ = self.mode.source(r)
        mass = bg.radius * self.solution.sol(bg.radius)[2]
        energy = np.dot(weights, 4 * np.pi * r**2 * g["alpha"] / np.sqrt(g["f"]) * rho)
        rhs = self.equation(r, y) + self.singular @ y / r
        B, Bp = bg.central * y[0], bg.central * y[1]
        dW = 2 * g["Omega"]**2 * (g["A"] * B + g["A"]**2 * y[3])
        dG = -2 * y[2] * g["Ap"]**2 + 2 * g["f"] * g["Ap"] * Bp
        dV = 2 * g["A"] * g["Vs"] * B
        drho, dpr = (dW + dG + dV) / 2, (dW + dG - dV) / 2
        vp = -dy[3]
        mass_residual = y[2] + r * dy[2] - 4 * np.pi * r**2 * (drho + rho)
        lapse_residual = vp - (1 + 8 * np.pi * r**2 * g["pr"]) * y[2] / (r * g["f"]**2)
        lapse_residual -= 4 * np.pi * r * (dpr + pr) / g["f"]
        charge = np.dot(weights, rhs[4])
        Vss = -8 / bg.scale**2 + 24 * g["A"]**2 / bg.scale**4
        k = y[2] / g["f"]
        kp = dy[2] / g["f"] - y[2] * g["fp"] / g["f"]**2
        kg_residual = g["f"] * bg.central * dy[1] + (
            2 * g["f"] / r + g["f"] * g["nup"] + g["fp"] / 2) * Bp
        kg_residual += (g["Omega"]**2 - g["Vs"] - 2 * g["A"]**2 * Vss) * B
        kg_residual += g["f"] * (vp - kp) * g["Ap"]
        kg_residual += (2 * g["Omega"]**2 * y[3] + 2 * k * (g["Omega"]**2 - g["Vs"])) * g["A"]
        background_nupp = -bg.sol(r, 2)[3]
        vpp = -self.solution.sol(r, 2)[3]
        angular = -2 * y[2] * (background_nupp + g["nup"]**2 + g["nup"] / r)
        angular += g["f"] * (vpp + (2 * g["nup"] + 1 / r) * vp)
        angular -= dy[2] * (g["nup"] + 1 / r)
        angular += g["fp"] * vp / 2
        angular_residual = angular / (8 * np.pi) - (dW - dG - dV) / 2 - pt
        action = action_data(bg, self.mode.ell,
                             v=lambda r: self.metric(r)[1],
                             k=lambda r: self.metric(r)[0] / (r * bg.fields(r)["f"]))
        completed_action = action_data(bg, self.mode.ell,
                                       v=lambda r: self.metric(r, completed=True)[1],
                                       k=lambda r: self.metric(r, completed=True)[0] / (r * bg.fields(r)["f"]))
        a, _, c = action["turning_points"]
        interval = np.linspace(a, c, 1001)
        x, v = self.metric(interval)
        step = 1e-5
        raw_vprime = (self.metric(interval + step)[1] - self.metric(interval - step)[1]) / (2 * step)
        return {"nodes": int(self.solution.x.size),
                "max_rms_residual": float(np.max(self.solution.rms_residuals)),
                "added_mass": float(mass), "axial_energy": float(energy),
                "mass_energy_error": float(abs(mass - energy)), "charge_integral": float(charge),
                "central_amplitude_response": float(bg.central * self.solution.sol(0)[0]),
                "relative_scalar_frequency_response": float(self.dlogomega),
                "max_ODE_residual": float(np.max(np.abs(dy - rhs))),
                "max_mass_equation_residual": float(np.max(np.abs(mass_residual))),
                "max_lapse_equation_residual": float(np.max(np.abs(lapse_residual))),
                "max_KG_equation_residual": float(np.max(np.abs(kg_residual))),
                "max_angular_Einstein_residual": float(np.max(np.abs(angular_residual))),
                "minimum_enclosed_mass_response_on_trapping_interval": float(np.min(x)),
                "minimum_raw_lapse_derivative_on_trapping_interval_sampled": float(np.min(raw_vprime)),
                "minimum_completed_lapse_derivative_on_trapping_interval": float(np.min(-self.solution.sol(interval, 1)[3])),
                "relative_light_ring_height_response": float(self.metric(bg.ru)[1] - self.metric(bg.rs)[1]),
                "action": action, "completed_action": completed_action}


if __name__ == "__main__":
    import json
    from trapped_axial import Background, TrappedAxial
    background = Background()
    for ell in (50000, 100000):
        mode = TrappedAxial(background, ell)
        response = MeanResponse(background, mode)
        print(json.dumps({"ell": ell, "response": response.diagnostics()}, indent=2))
