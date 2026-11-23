"""Static zero-mode diagnostics by transported regular and exterior subspaces.

Fourth-order Magnus propagation with QR keeps exponentially growing scalar
solutions from erasing the admissible subspaces. These are floating-point
diagnostics, not interval certificates for the exact asymptotic problem.
"""

import numpy as np
from scipy.integrate import solve_bvp
from scipy.linalg import expm, null_space
from scipy.optimize import brentq

from boson_star import SINGULAR, potential
from mean_response import background_jacobian, tail_variation
from trapped_axial import gauss_mesh


SINGULAR5 = np.zeros((5, 5))
SINGULAR5[:4, :4] = SINGULAR


def static_matrix(background, radius, regular=False):
    r = np.atleast_1d(radius)
    g = background.fields(r)
    matrix = np.zeros((r.size, 5, 5))
    matrix[:, :4, :4] = np.moveaxis(background_jacobian(background, r), -1, 0)
    prefactor = 4 * np.pi * r**2 * g["Omega"] / np.sqrt(g["f"])
    matrix[:, 4, 0] = prefactor * 2 * g["A"] * background.central
    matrix[:, 4, 2] = prefactor * g["A"]**2 / g["f"]
    matrix[:, 4, 3] = prefactor * g["A"]**2
    if not regular:
        if np.any(r == 0):
            raise ValueError("Use the regular matrix at the center.")
        matrix += SINGULAR5 / r[:, None, None]
    return matrix


def central_basis(background, radius):
    """Regular central Taylor data for delta a(0)=1 and delta zeta(0)=1."""
    A0 = background.central
    Omega2 = np.exp(2 * background.sol(0)[3])
    _, Vs = potential(A0, background.scale)
    Vss = -8 / background.scale**2 + 24 * A0**2 / background.scale**4
    drho = A0**2 * np.array([Omega2 + Vs, Omega2])
    dpr = A0**2 * np.array([Omega2 - Vs, Omega2])
    second_a = np.array([Vs + 2 * A0**2 * Vss - Omega2, -2 * Omega2]) / 3
    basis = np.zeros((5, 2))
    basis[0] = np.array([1., 0.]) + second_a * radius**2 / 2
    basis[1] = second_a * radius
    basis[2] = 4 * np.pi / 3 * drho * radius**2
    basis[3] = np.array([0., 1.]) - 2 * np.pi * (drho / 3 + dpr) * radius**2
    basis[4] = 4 * np.pi / 3 * np.sqrt(Omega2) * A0**2 * np.array([2., 1.]) * radius**3
    return basis


def background_charge(background):
    r, weights = gauss_mesh(background.sol.x)
    g = background.fields(r)
    return float(np.dot(weights, 4 * np.pi * r**2 * g["Omega"] * g["A"]**2 / np.sqrt(g["f"])))


def bottom_barrier_action(background, order=160):
    """Barrier action evaluated at the stable-light-ring potential minimum."""
    outer = brentq(lambda r: background.fields(r)["U"] - background.Us, background.ru, background.radius)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    theta = (nodes + 1) * np.pi / 4
    r = background.rs + (outer - background.rs) * np.sin(theta)**2
    g = background.fields(r)
    measure = weights * np.pi / 4 * (outer - background.rs) * np.sin(2 * theta)
    return float(np.dot(measure, np.sqrt(np.maximum(g["U"] - background.Us, 0)) / g["p"]))


def transport_subspace(background, mesh, initial, charge_scale):
    q = np.linalg.qr(initial, mode="reduced")[0]
    scaling = np.array([1., 1., 1., 1., 1 / charge_scale])
    step = np.diff(mesh)
    middle = (mesh[:-1] + mesh[1:]) / 2
    r1 = middle - step / (2 * np.sqrt(3))
    r2 = middle + step / (2 * np.sqrt(3))
    A1, A2 = static_matrix(background, r1), static_matrix(background, r2)
    A1 *= scaling[None, :, None] / scaling[None, None, :]
    A2 *= scaling[None, :, None] / scaling[None, None, :]
    for h, first, second in zip(step, A1, A2):
        exponent = h * (first + second) / 2 - np.sqrt(3) * h**2 * (first @ second - second @ first) / 12
        q = np.linalg.qr(expm(exponent) @ q, mode="reduced")[0]
    return q


def matching_diagnostics(background, resolution=1, center_radius=.002, fixed_charge=True):
    if resolution not in (1, 2, 3):
        raise ValueError("Only the three bounded resolutions are implemented.")
    rs, R = background.rs, background.radius
    charge = background_charge(background)
    scaling = np.array([1., 1., 1., 1., 1 / charge])
    inner_mesh = np.unique(np.r_[np.geomspace(center_radius, .1, 20 * resolution + 1),
                                np.linspace(.1, rs, 500 * resolution + 1)])
    outer_mesh = np.linspace(R, rs, 800 * resolution + 1)
    if inner_mesh.size + outer_mesh.size > 4000:
        raise RuntimeError("Static propagation exceeds the 4000-point limit.")
    initial = central_basis(background, center_radius)
    if not fixed_charge:
        initial = initial[:, 1:]
    inner = transport_subspace(background, inner_mesh, scaling[:, None] * initial, charge)
    boundary_rows = np.zeros((2 if fixed_charge else 1, 5))
    boundary_rows[0] = [tail_variation(background, np.eye(5)[j]) / scaling[j] for j in range(5)]
    if fixed_charge:
        boundary_rows[1, 4] = 1
    outer = transport_subspace(background, outer_mesh, null_space(boundary_rows), charge)
    matching = np.column_stack((-inner, outer))
    singular_values = np.linalg.svd(matching, compute_uv=False)
    g = background.fields(rs)
    z = g["alpha"] / np.sqrt(g["f"])
    jump = np.array([0, background.sol(rs)[1] / g["f"], 1, 0, 0]) / (rs * z)
    coefficients = np.linalg.solve(matching, scaling * jump)
    dimension = inner.shape[1]
    left = inner @ coefficients[:dimension] / scaling
    right = outer @ coefficients[dimension:] / scaling
    return {"resolution": resolution, "center_radius": center_radius, "fixed_charge": fixed_charge,
            "outer_radius": R, "radial_points": int(inner_mesh.size + outer_mesh.size),
            "charge_scale": charge, "singular_values": singular_values.tolist(),
            "minimum_singular_value": float(singular_values[-1]),
            "matching_inverse_norm": float(1 / singular_values[-1]),
            "absolute_determinant": float(abs(np.linalg.det(matching))),
            "left_state_at_light_ring": left.tolist(), "right_state_at_light_ring": right.tolist(),
            "interface_residual": float(np.max(np.abs(right - left - jump)))}


class BranchTangent:
    """Equilibrium tangent normalized by delta a(0)=1, with charge unconstrained."""
    def __init__(self, background, tolerance=1e-7):
        self.background = background
        r = np.unique(np.r_[np.linspace(0, background.radius, 401), .002 * 2**np.arange(6)])
        def rhs(r, y):
            return np.einsum("nij,jn->in", static_matrix(background, r, regular=True), y)
        def bc(left, right):
            return np.array([left[0] - 1, left[1], left[2], left[4], tail_variation(background, right)])
        self.solution = solve_bvp(rhs, bc, r, np.zeros((5, r.size)), S=SINGULAR5,
                                  tol=tolerance, max_nodes=4000)
        if not self.solution.success:
            raise RuntimeError("Branch tangent failed within 4000 points: " + self.solution.message)
        end = self.solution.sol(background.radius)
        self.charge_response = end[4]
        self.mass_response = background.radius * end[2]
        self.dlogomega = end[3] - end[2] / background.fields(background.radius)["f"]

    def metric(self, r):
        state = self.solution.sol(r)
        return np.asarray(r)*state[2], self.dlogomega-state[3]

    def diagnostics(self):
        bg = self.background
        return {"radial_points": int(self.solution.x.size),
                "max_rms_residual": float(np.max(self.solution.rms_residuals)),
                "normalization": "delta a(0)=1, hence delta A(0)=A0",
                "dN_dA0": float(self.charge_response / bg.central),
                "dM_dA0": float(self.mass_response / bg.central),
                "domega_dA0": float(bg.omega * self.dlogomega / bg.central),
                "first_law_error": float(abs(self.mass_response - bg.omega * self.charge_response))}


if __name__ == "__main__":
    import json
    from trapped_axial import Background
    background = Background()
    for resolution in (1, 2):
        print(json.dumps(matching_diagnostics(background, resolution), indent=2))
    print(json.dumps(BranchTangent(background).diagnostics(), indent=2))
