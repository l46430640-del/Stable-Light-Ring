"""Local angular response, exact angular sum, and conditional optical bounds."""

from math import isfinite

import numpy as np
from numpy.polynomial import Legendre
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from finite_width_axial import OrbitSource
from polar_factorization import dipole_coefficients
from polar_static import angular_parameters
from trapped_axial import gauss_mesh


def local_optical(source, r):
    r = np.atleast_1d(r)
    rho, pr, _, _ = source.fields(r)
    return -8*np.pi*r*r*(2*rho-pr), -8*np.pi*r*r*(rho-2*pr)


def orbit_quadrature(probe, source, order, barrier=False):
    left, right = (probe.b, probe.c) if barrier else (probe.a, probe.b)
    knots = source.knots[(source.knots > left) & (source.knots < right)]
    breaks = np.r_[0., 2/np.pi*np.arcsin(np.sqrt((knots-left)/(right-left))), 1.]
    t, weights = gauss_mesh(np.unique(breaks), order)
    if len(t) > 4000:
        raise ValueError("The orbit quadrature exceeds 4000 points.")
    r, _, gap, ratio = probe.mapped(t, barrier)
    return r, weights*ratio/probe.background.fields(r)["p"], gap


def leading_action(source, fraction, order=240):
    probe = OrbitSource(source.background, fraction, order=order)
    r, weights, gap = orbit_quadrature(probe, source, order)
    v, k = local_optical(source, r)
    vstar = float(weights@(v-k*gap/probe.energy)/sum(weights))
    r, weights, gap = orbit_quadrature(probe, source, order, barrier=True)
    v, k = local_optical(source, r)
    return float(weights@(k*gap+probe.energy*(v-vstar)))


def remainder_source_norms(source, order=24):
    knots = np.unique(np.r_[np.linspace(source.a, source.b, 101), source.knots])
    r, weights = gauss_mesh(knots, order)
    if len(r) > 4000:
        raise ValueError("The quadrature exceeds 4000 points.")
    rho, pr, D, Dp = source.fields(r)
    c = dipole_coefficients(source.background, r)
    n = c["g"]["nup"]
    prp = -n*(rho+pr)-(3*pr-rho)/r
    rhop = prp+Dp/(np.pi*r*r)-2*D/(np.pi*r**3)
    b = np.column_stack((8*np.pi*r*r*rho, np.zeros_like(r)))
    bp = np.column_stack((8*np.pi*(2*r*rho+r*r*rhop), np.zeros_like(r)))
    T = bp-c["S"]*b-8*c["a"]*D[:, None]
    J = np.einsum("nij,nj->ni", c["M"], b)
    return {"T": float(np.sqrt(weights@(c["P"]*np.sum(T*T, axis=1)))),
            "J": float(np.sqrt(weights@(c["P"]/c["W"]*np.sum(J*J, axis=1)))),
            "quadrature_points": len(r)}


def angular_sum(z):
    z = np.asarray(z)
    if np.any(~np.isfinite(z)) or np.any(abs(z) > 1):
        raise ValueError("Use a normal cosine in [-1,1].")
    return .5-2/np.pi*np.sqrt(1-z*z)


def resummed_kernel(D0, harmonics, leading):
    if not harmonics or set(harmonics) != set(range(2, max(harmonics)+1, 2)):
        raise ValueError("Supply every even degree from two to the cutoff.")
    coefficients = np.zeros(max(harmonics)+1)
    coefficients[0] = D0
    for degree, value in harmonics.items():
        L, k, _ = angular_parameters(degree)
        coefficients[degree] = (2*degree+1)*(value-leading*k*k/(L-2))
    polynomial = Legendre(coefficients)
    return lambda z: polynomial(z)+leading*angular_sum(z)


def sampled_minimum(function, count=2001):
    if not 101 <= count <= 4000:
        raise ValueError("Use 101 to 4000 angular samples.")
    z = np.sin(np.linspace(0, np.pi/2, count))
    values = function(z)
    candidates = [(float(values[0]), 0.), (float(values[-1]), 1.)]
    for i in np.flatnonzero((values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:]))+1:
        result = minimize_scalar(function, bounds=(z[i-1], z[i+1]), method="bounded")
        if not result.success:
            raise RuntimeError("Angular minimum refinement did not converge.")
        candidates.append((float(result.fun), float(result.x)))
    value, point = min(candidates)
    return {"minimum": value, "normal_cosine": point, "angular_points": count,
            "scope": "Floating-point sampling and local refinement; not a global certificate."}


class OpticalWeights:
    """Compact cumulative weights; square-root endpoints use the orbit angle."""
    def __init__(self, background, fraction, count=801):
        if not 101 <= count <= 1601:
            raise ValueError("Use 101 to 1601 angle points per orbit interval.")
        self.background = background
        self.probe = OrbitSource(background, fraction, order=240)
        self.t = np.linspace(0, 1, count)
        self.integrals = []
        for barrier in (False, True):
            r, _, _, ratio = self.probe.mapped(self.t, barrier)
            g = background.fields(r)
            self.integrals.append([CubicSpline(self.t, ratio/g["p"]*factor).antiderivative()
                                   for factor in (np.ones_like(r), g["U"])])
        self.Z = float(self.integrals[0][0](1)-self.integrals[0][0](0))
        self.B = float(self.integrals[1][0](1)-self.integrals[1][0](0))
        self.count = count

    def H(self, radius):
        r = np.atleast_1d(radius)
        if np.any(r < 0) or np.any(r > self.background.radius):
            raise ValueError("Use radii inside the represented background.")
        p = self.probe
        result = np.zeros_like(r, dtype=float)
        for index, (left, right) in enumerate(((p.a, p.b), (p.b, p.c))):
            t = 2/np.pi*np.arcsin(np.sqrt(np.clip((r-left)/(right-left), 0, 1)))
            integral = self.integrals[index][1]
            result += (self.B/self.Z if index == 0 else -1)*(integral(1)-integral(t))
        return result

    def constants(self, order=24):
        p, bg = self.probe, self.background
        r, weights = self.radial_quadrature(order)
        c = dipole_coefficients(bg, r)
        Hnorm = np.sqrt(weights@(self.H(r)**2*np.sum(c["a"]**2, axis=1)/(c["P"]*c["W"])))
        r = np.linspace(p.a, bg.radius, 2401)
        c = dipole_coefficients(bg, r)
        values = (1/(r*c["g"]["alpha"]), np.maximum(-c["S"][:, 0], 0)/(c["P"]*c["W"]))
        maxima = []
        for index, profile in enumerate(values):
            i = int(np.argmax(profile))
            def f(x):
                cc = dipole_coefficients(bg, [x])
                return float(1/(x*cc["g"]["alpha"][0]) if index == 0 else
                             max(-cc["S"][0, 0], 0)/(cc["P"][0]*cc["W"][0]))
            result = minimize_scalar(lambda x: -f(x), bounds=(r[max(0, i-1)], r[min(len(r)-1, i+1)]), method="bounded")
            if not result.success:
                raise RuntimeError("Optical constant refinement did not converge.")
            maxima.append(max(float(profile.max()), float(-result.fun)))
        return {"L": maxima[0], "B_S": maxima[1], "H": float(Hnorm),
                "V": 4*p.energy*self.B, "barrier_weight": self.B,
                "well_weight": self.Z, "angle_points_per_interval": self.count,
                "quadrature_points": len(weights),
                "scope": "Numerical finite-domain constants; L and B_S also require exterior control."}

    def radial_quadrature(self, order=24, knots=None):
        p = self.probe
        knots = np.array([]) if knots is None else np.asarray(knots)
        center_knots = np.unique(np.r_[np.linspace(0, p.a, 81), knots[(knots > 0) & (knots < p.a)]])
        center, cw = gauss_mesh(center_knots, order)
        radii, quadrature = [center], [cw]
        for barrier in (False, True):
            left, right = (p.b, p.c) if barrier else (p.a, p.b)
            selected = knots[(knots > left) & (knots < right)]
            breaks = np.r_[0., 2/np.pi*np.arcsin(np.sqrt((selected-left)/(right-left))), 1.]
            t, weights = gauss_mesh(np.unique(breaks), 10*order)
            r, dr, _, _ = p.mapped(t, barrier)
            radii.append(r)
            quadrature.append(weights*dr)
        r, q = np.concatenate(radii), np.concatenate(quadrature)
        if len(r) > 4000:
            raise ValueError("The optical quadrature exceeds 4000 points.")
        return r, q


def conditional_tail_bound(cutoff, negative_bound, T, J, L, B_S, H, V):
    """Uniform remainder tail; every supplied constant must bound the exact one."""
    if not isinstance(cutoff, (int, np.integer)) or cutoff < 2 or cutoff % 2:
        raise ValueError("Use an even cutoff at least two.")
    cutoff = int(cutoff)
    if any(not isfinite(x) or x < 0 for x in (negative_bound, T, J, L, B_S, H, V)):
        raise ValueError("Bounds must be finite and nonnegative.")
    first = cutoff+2
    gamma = 1-negative_bound/(first*(first+1)-2)
    if gamma <= 0:
        raise ValueError("The first omitted degree must have positive coercivity.")
    source = np.sqrt(T*T+J*J/(gamma*first*first))
    A1 = V*np.sqrt(L/np.sqrt(gamma)+2*B_S/(gamma*first))
    A2 = 2*H/np.sqrt(gamma)
    return float(source*(2*A1/(3*cutoff**1.5)+A2/(2*cutoff**2)))
