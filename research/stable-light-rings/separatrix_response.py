"""Coefficient of the inverse-logarithmic action response near the barrier top.

All functions here evaluate the represented background in floating point.
The static response at the unstable light ring is computed independently of
the finite-width energy scan.
"""

import numpy as np
from scipy.optimize import brentq

from trapped_axial import gauss_mesh


def upper_well_quadrature(background, order=160, breakpoints=()):
    """Return r, q dr/sqrt(Uu-U), Uu-U and the limiting well geometry.

    The measure has infinite total mass. It is used only with numerators
    vanishing at the unstable orbit, including the finite-part moments.
    """
    bg, ru = background, background.ru
    inner = brentq(lambda r: bg.fields(r)["U"] - bg.Uu, bg.sol.x[1], bg.rs, xtol=5e-14)
    g_u = bg.fields(ru)
    # At a circular orbit, U''=2 U H'/(r f) and H'=K/r.
    K = 8 * np.pi * ru**2 * g_u["Omega"]**2 * g_u["A"]**2 - 1
    curvature = -bg.Uu * K / (ru**2 * g_u["f"])
    if curvature <= 0:
        raise ValueError("The outer circular orbit must be a nondegenerate maximum.")
    width = ru - inner
    cuts = np.asarray(breakpoints, dtype=float)
    if np.any(~np.isfinite(cuts)) or np.any(cuts < inner) or np.any(cuts > ru):
        raise ValueError("Quadrature breaks must lie in the limiting well.")
    angles = np.unique(np.r_[0., np.sqrt((cuts-inner)/width), 1.])
    if order*(len(angles)-1) > 4000:
        raise ValueError("At most 4000 quadrature points are supported.")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    t = ((angles[:-1, None]+angles[1:, None])/2+np.diff(angles)[:, None]*nodes/2).ravel()
    weights = (np.diff(angles)[:, None]*weights/2).ravel()
    r = inner + width * t**2
    length = -width * (1 - t) * (1 + t)
    gap = bg.Uu - bg.fields(r)["U"]
    near = np.abs(length) < .002
    qnodes, qweights = np.polynomial.legendre.leggauss(8)
    near_inner = width * t**2 < .001
    if np.any(near_inner):
        distance = width * t[near_inner]**2
        points = inner + distance[:, None] * (qnodes + 1) / 2
        g = bg.fields(points.ravel())
        Up = g["U"] * (-2 * bg.sol(points.ravel(), 1)[3] - 2 / points.ravel())
        gap[near_inner] = -distance / 2 * (Up.reshape(points.shape) @ qweights)
    if np.any(near):
        points = ru + length[near, None] * (qnodes + 1) / 2
        g = bg.fields(points.ravel())
        Up = 2 * g["U"] * (g["nup"] - 1 / points.ravel())
        gap[near] = -length[near] / 2 * (Up.reshape(points.shape) @ qweights)
    if np.any(gap <= 0):
        raise ValueError("The limiting well gap must be positive.")
    g = bg.fields(r)
    measure = weights * 2 * width * t / (g["p"] * np.sqrt(gap))
    return r, measure, gap, float(inner), float(curvature)


def upper_well_data(background, response, order=160):
    if response.light_ring != "unstable":
        raise ValueError("The separatrix coefficient needs the unstable-ring response.")
    bg, ru = background, background.ru
    r, measure, gap, inner, curvature = upper_well_quadrature(bg, order)
    g, g_u = bg.fields(r), bg.fields(ru)
    length = r-ru
    near = np.abs(length) < .002
    qnodes, qweights = np.polynomial.legendre.leggauss(8)
    x, v = response.metric(r, side="left")
    vu = response.metric(ru, side="left")[1]
    difference = vu - v
    if np.any(near):
        points = ru + length[near, None] * (qnodes + 1) / 2
        vp = -response.state(points.ravel(), 1, side="left")[3]
        difference[near] = -length[near] / 2 * (vp.reshape(points.shape) @ qweights)
    mass_part = float(np.dot(measure, x / (r * g["f"]) * gap / bg.Uu))
    lapse_part = float(np.dot(measure, difference))
    H = mass_part + lapse_part
    barrier_weight = np.pi / (g_u["p"] * np.sqrt(curvature))
    log_normalization = 1 / (2 * g_u["p"] * np.sqrt(curvature))
    sample = np.linspace(inner, ru, 401)
    y = response.state(sample, side="left")
    vp = -response.state(sample, 1, side="left")[3]
    return {"inner_turning_point_at_U_u": float(inner), "unstable_radius": float(ru),
            "curvature_minus_half_Upp": float(curvature), "well_action_at_U_u": float(np.dot(measure, gap)),
            "H": H, "H_mass_term": mass_part, "H_lapse_term": lapse_part,
            "barrier_weight_limit": float(barrier_weight),
            "logarithmic_normalization_coefficient": float(log_normalization),
            "Z_times_action_limit": float(barrier_weight * bg.Uu * H),
            "log_times_action_limit": float(2 * np.pi * bg.Uu * H),
            "inner_well_mass_ratio_min_sampled": float(np.min(y[2])),
            "inner_well_lapse_slope_min_sampled": float(np.min(vp))}


def interface_diagnostics(response):
    bg, ru = response.background, response.location
    left, right = response.state(ru, side="left"), response.state(ru)
    dl, dr = response.state(ru, 1, "left"), response.state(ru, 1)
    t, weights = gauss_mesh(response.solution.x, order=4)
    y, derivative = response.solution.sol(t), response.solution.sol(t, 1)
    rhs = response.equation(t, y)
    rhs[:5] += response.singular @ y[:5] / t
    charge = response.widths[0] * np.dot(weights, response.homogeneous(ru * t, y[:5])[4])
    charge += response.widths[1] * np.dot(weights, response.homogeneous(ru + response.widths[1] * t, y[5:])[4])
    jump = -dr[3] + dl[3]
    expected = 3 / (ru**2 * bg.fields(ru)["p"])
    return {"radial_points": int(2 * response.solution.x.size),
            "max_rms_residual": float(np.max(response.solution.rms_residuals)),
            "max_angle_ODE_residual": float(np.max(np.abs(derivative - rhs))),
            "jump_residual": float(np.max(np.abs(right - left - response.jump))),
            "lapse_slope_jump_error": float(abs(jump - expected)),
            "added_mass": float(bg.radius * response.state(bg.radius)[2]),
            "charge_integral": float(charge),
            "mass_response_left": float(ru * left[2]), "mass_response_right": float(ru * right[2]),
            "lapse_derivative_left": float(-dl[3]), "lapse_derivative_right": float(-dr[3])}
