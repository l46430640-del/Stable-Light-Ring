"""Finite-radius static response with charge eliminated by the first integral.

The endpoint rows define a reference boundary problem, not a certificate for
the regular center and asymptotically flat tail of an exact boson star.
"""

from fractions import Fraction

import numpy as np
from scipy.linalg import null_space
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from mpmath import iv

from boson_star import SINGULAR
from check_barrier_certificate import Jet, exact, matrix_formula, polynomial
from mean_response import tail_variation
from static_transversality import central_basis, static_matrix


WEIGHTS = np.array([1., 2., .4, .5])
SCALED_SINGULAR = SINGULAR * WEIGHTS[None, :] / WEIGHTS[:, None]


def endpoint_rows(background, left):
    center = central_basis(background, left)[:4] / WEIGHTS[:, None]
    Bleft = null_space(center.T).T
    R = background.radius
    g = background.fields(R)
    energy = np.array([-4 * np.pi * R * g["f"] * g["Ap"] * background.central, 0, 1., 0])
    tail = np.array([tail_variation(background, np.eye(4)[j]) for j in range(4)])
    raw = np.vstack((tail, energy)) * WEIGHTS[None, :]
    q, triangular = np.linalg.qr(raw.T, mode="reduced")
    return Bleft, q.T, np.linalg.solve(triangular.T, [0., 1 / R])


def coefficient_jets(background, radius, piece, regular=False):
    r = Jet(0, [radius, iv.mpf(1), iv.mpf(0), iv.mpf(0)])
    argument = r - exact(float(background.sol.x[piece]))
    y = [polynomial(background.sol.c[:, piece, j], argument) for j in range(4)]
    matrix = matrix_formula(r, y, iv.mpf(str(background.central)), iv.mpf(str(background.scale)), regular=regular)
    # Binary weights define the exact scaling used by the discrete operator.
    return [[entry * (exact(float(WEIGHTS[j])) / exact(float(WEIGHTS[i])))
             for j, entry in enumerate(row)] for i, row in enumerate(matrix)]


def local_polynomials(background, mesh, degree=12):
    """Approximate fundamental matrices; residual bounds must validate them."""
    coefficients, regular_coefficients, transitions = [], [], []
    nodes = np.cos(np.pi * (np.arange(9) + .5) / 9)
    for left, right in zip(mesh[:-1], mesh[1:]):
        center = (Fraction(float(left)) + Fraction(float(right))) / 2
        half = (Fraction(float(right)) - Fraction(float(left))) / 2
        sample = static_matrix(background, float(center) + float(half) * nodes, regular=True)[:, :4, :4]
        sample *= WEIGHTS[None, None, :] / WEIGHTS[None, :, None]
        chebyshev = np.polynomial.chebyshev.chebfit(nodes, sample.reshape(9, 16), 8)
        powers = np.zeros((9, 16))
        for j in range(16):
            column = np.polynomial.chebyshev.cheb2poly(chebyshev[:, j])
            powers[:len(column), j] = column
        G = powers.reshape(9, 4, 4)
        A = np.array([SCALED_SINGULAR * (-float(half)/float(center))**k * float(half)/float(center)
                      for k in range(degree)])
        A[:len(G)] += G * float(half)
        P = np.zeros((degree+1, 4, 4))
        P[0] = np.eye(4)
        for k in range(1, degree+1):
            P[k] = sum(A[j] @ P[k-1-j] for j in range(min(k, len(A)))) / k
        minus = sum((-1)**k * P[k] for k in range(degree+1))
        plus = np.sum(P, axis=0)
        coefficients.append(P)
        regular_coefficients.append(G)
        transitions.append(np.linalg.solve(minus.T, plus.T).T)
    return np.array(coefficients), np.array(regular_coefficients), np.array(transitions)


def discrete_inverse(mesh, transitions, left, right):
    assert transitions.shape == (len(mesh)-1, 4, 4)
    assert left.ndim == right.ndim == 2 and left.shape[1] == right.shape[1] == 4
    assert len(left)+len(right) == 4 and 0 < len(left) < 4
    assert all(np.all(np.isfinite(array)) for array in (mesh, transitions, left, right))
    if len(mesh) > 450:
        raise RuntimeError("The dense inverse diagnostic is capped at 450 radial nodes.")
    N = len(mesh) - 1
    size = 4 * len(mesh)
    rows, cols, values = [], [], []
    for k, T in enumerate(transitions):
        for i in range(4):
            rows.append(4*k+i)
            cols.append(4*(k+1)+i)
            values.append(1.)
            for j in range(4):
                rows.append(4*k+i)
                cols.append(4*k+j)
                values.append(-T[i, j])
    for i in range(len(left)):
        for j in range(4):
            rows.append(4*N+i)
            cols.append(j)
            values.append(left[i, j])
    for i in range(len(right)):
        for j in range(4):
            rows.append(4*N+len(left)+i)
            cols.append(4*N+j)
            values.append(right[i, j])
    operator = coo_matrix((values, (rows, cols)), shape=(size, size)).tocsc()
    factor = splu(operator)
    inverse = np.empty((size, size))
    for start in range(0, size, 64):
        stop = min(start+64, size)
        rhs = np.zeros((size, stop-start))
        rhs[np.arange(start, stop), np.arange(stop-start)] = 1
        inverse[:, start:stop] = factor.solve(rhs)
    return inverse, factor


def build_discrete_inverse(background):
    mesh = np.unique(np.r_[np.geomspace(.002, 1, 121), np.linspace(1, 12, 61),
                            np.linspace(12, 20, 161), np.linspace(20, background.radius, 61), background.rs])
    left, right, right_rhs = endpoint_rows(background, mesh[0])
    P, G, transitions = local_polynomials(background, mesh)
    inverse, factor = discrete_inverse(mesh, transitions, left, right)
    size = len(inverse)
    rs_index = int(np.flatnonzero(mesh == background.rs)[0])
    g = background.fields(background.rs)
    zs = g["alpha"] / np.sqrt(g["f"])
    jump = np.array([0, background.sol(background.rs)[1] / g["f"], 1, 0]) / (background.rs * zs)
    rhs = np.zeros(size)
    rhs[4*(rs_index-1):4*rs_index] = jump / WEIGHTS
    rhs[-2:] = right_rhs
    state = factor.solve(rhs).reshape(-1, 4) * WEIGHTS
    row_weights = np.r_[np.repeat(np.diff(mesh), 4), np.ones(4)]
    report = {"radial_nodes": len(mesh), "matrix_dimension": size,
              "inverse_storage_MiB": inverse.nbytes / 2**20,
              "inverse_infinity_norm": float(np.max(np.sum(np.abs(inverse), axis=1))),
              "weighted_forcing_gain": float(np.max(np.abs(inverse) @ row_weights)),
              "right_interface_state": state[rs_index].tolist()}
    return mesh, P, G, transitions, left, right, inverse, state, rhs, report


if __name__ == "__main__":
    import json
    from trapped_axial import Background
    *_, report = build_discrete_inverse(Background())
    print(json.dumps(report, indent=2))
