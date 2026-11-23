"""Dipole deformation of the static polar operator, not a dynamical energy."""

from math import comb, isfinite, sqrt

import numpy as np

from polar_static import coefficients


def dipole_coefficients(background, radius):
    r = np.atleast_1d(np.asarray(radius, dtype=float))
    if r.ndim != 1 or np.any(~np.isfinite(r)) or np.any(r <= 0) or np.any(r > background.radius):
        raise ValueError("Use positive radii inside the represented background.")
    g, d, _, np_, App, C, Av, Au = coefficients(background, r)
    if np.any(g["nup"] <= 0) or np.any(g["Ap"] >= 0):
        raise ValueError("The diagonal dipole deformation requires n>0 and A'<0.")
    a = np.column_stack((g["nup"], np.sqrt(4*np.pi)*g["Ap"]))
    ap = np.column_stack((np_, np.sqrt(4*np.pi)*App))
    G = g["nup"]-g["fp"]/(2*g["f"])
    S = ap/a-G[:, None]
    M = np.zeros((len(r), 2, 2))
    M[:, 0, 1] = M[:, 1, 0] = np.sqrt(4*np.pi)*C
    M[:, 0, 0] = -M[:, 0, 1]*a[:, 1]/a[:, 0]
    M[:, 1, 1] = -M[:, 0, 1]*a[:, 0]/a[:, 1]
    Q2 = M.copy()
    Q2[:, 0, 0], Q2[:, 1, 1] = Av, Au
    P, W = r*r*g["p"], 1/(r*r*g["f"])
    return {"g": g, "d": d, "P": P, "W": W, "a": a, "ap": ap,
            "S": S, "M": M, "Q2": Q2,
            "negative_bound": np.maximum(-np.trace(M, axis1=1, axis2=2), 0)/W}


def exterior_negative_trace(s, f, mass_pressure, d2, d4, t, omega2, vs):
    """-r^2 f tr(M); d_j=4*pi*r^j*A^2, t=A'/A, omega2=Omega^2.

    No division by the exponentially small scalar amplitude is needed.
    Inputs may be symbolic expressions or interval scalars.
    """
    h = 2*omega2-vs
    return (4*mass_pressure**2*s**2/f+2*mass_pressure*h/(f*t)
            +4*d2*f*t*t+2*d4*f*t*h/mass_pressure)


def conditional_response_bound(degree, negative_bound, density_norm, anisotropy_norm):
    """Energy and weighted L2 bounds, conditional on a GLOBAL negative bound."""
    if not isinstance(degree, (int, np.integer)) or degree < 2 or degree % 2:
        raise ValueError("Use an even integer degree at least two.")
    degree = int(degree)
    values = (negative_bound, density_norm, anisotropy_norm)
    if any(not isfinite(x) or x < 0 for x in values):
        raise ValueError("All constants must be finite and nonnegative.")
    L = degree*(degree+1)
    mu = L-2-negative_bound
    if mu <= 0:
        raise ValueError("The assumed global bound must give positive coercivity.")
    k = comb(degree, degree//2)/2**degree
    energy_norm = k*sqrt(density_norm**2/mu+64*anisotropy_norm**2/(L-2)**2)
    return energy_norm, energy_norm/sqrt(mu)
