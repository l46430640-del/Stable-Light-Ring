"""Optical dual norm from compact fluxes near the turning points."""

from math import isfinite

import numpy as np

from polar_factorization import dipole_coefficients
from trapped_axial import gauss_mesh


class OpticalDualBound:
    """Piecewise constant endpoint averages remove the non-L2 weights."""

    def __init__(self, weights, mu, scale=2., deformation=None):
        if not isfinite(mu) or mu <= 0 or not isfinite(scale) or scale <= 0:
            raise ValueError("The coercivity and layer scale must be positive and finite.")
        self.weights, self.mu, self.scale = weights, float(mu), float(scale)
        if deformation is not None and not callable(deformation):
            raise ValueError("Supply a callable matrix deformation.")
        self.deformation = deformation
        self.layers = []
        p, bg = weights.probe, weights.background
        for barrier, (left, right) in enumerate(((p.a, p.b), (p.b, p.c))):
            width = right-left
            endpoints = np.array([left, right])
            length = np.minimum(width/2, scale*endpoints*np.sqrt(bg.fields(endpoints)["f"])/np.sqrt(mu))
            cuts = np.array([0., 2/np.pi*np.arcsin(np.sqrt(length[0]/width)),
                             2/np.pi*np.arcsin(np.sqrt(1-length[1]/width)), 1.])
            if cuts[1] <= 0 or cuts[2] >= 1:
                raise ValueError("An endpoint layer is below the represented angular precision.")
            if cuts[1] >= cuts[2]-1e-14:
                cuts[1:3] = .5
            factor = 2*p.energy*(1. if barrier else -weights.B/weights.Z)
            primitive = weights.integrals[barrier][0]
            for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
                if a == b:
                    continue
                ra, rb = p.mapped([a, b], bool(barrier))[0]
                if rb <= ra:
                    raise ValueError("An endpoint layer is below the represented radial precision.")
                average = factor*float(primitive(b)-primitive(a))/(rb-ra) if i != 1 else None
                self.layers.append((bool(barrier), a, b, ra, average, factor))

    def quadrature(self, order=24):
        if (isinstance(order, bool) or not isinstance(order, (int, np.integer))
                or not 8 <= order <= 32):
            raise ValueError("Use a quadrature order from 8 to 32.")
        p = self.weights.probe
        r, q = gauss_mesh(np.linspace(0, p.a, 81), int(order))
        radii, quadrature = [r], [q]
        flux, regular = [np.zeros_like(r)], [np.zeros_like(r)]
        for barrier, a, b, ra, average, factor in self.layers:
            breaks = np.array([a, b])
            if self.deformation is not None:
                left, right = (p.b, p.c) if barrier else (p.a, p.b)
                extra = np.asarray(getattr(self.deformation, "breakpoints", ()))
                extra = extra[(extra > left) & (extra < right)]
                angles = 2/np.pi*np.arcsin(np.sqrt((extra-left)/(right-left)))
                breaks = np.unique(np.r_[breaks, angles[(angles > a) & (angles < b)]])
            t, q = gauss_mesh(breaks, 5*int(order))
            r, dr, _, ratio = p.mapped(t, barrier)
            if average is None:
                j = factor*ratio/(self.weights.background.fields(r)["p"]*dr)
                R = np.zeros_like(r)
            else:
                primitive = self.weights.integrals[int(barrier)][0]
                j = np.full_like(r, average)
                R = factor*(primitive(t)-primitive(a))-average*(r-ra)
            radii.append(r)
            quadrature.append(q*dr)
            flux.append(R)
            regular.append(j)
        r, q, R, j = [np.concatenate(values) for values in (radii, quadrature, flux, regular)]
        if len(r) > 4000:
            raise ValueError("The scalar quadrature exceeds 4000 radial points.")
        return r, q, R, j

    def evaluate(self, order=24):
        r, q, R, j = self.quadrature(order)
        c = dipole_coefficients(self.weights.background, r)
        F = -2*self.weights.H(r)[:, None]*c["a"]
        F[:, 0] += j
        if self.deformation is None:
            F[:, 0] -= R*c["S"][:, 0]
        else:
            T = np.asarray(self.deformation(r))
            if T.shape != (len(r), 2, 2) or not np.all(np.isfinite(T)):
                raise ValueError("The deformation must return finite two-by-two matrices.")
            F -= R[:, None]*T[:, 0, :]
        derivative_part = float(q@(R*R/c["P"]))
        field_part = float(q@(np.sum(F*F, axis=1)/(c["P"]*c["W"])))
        bound = np.sqrt(derivative_part+field_part/self.mu)
        return {"optical_dual_bound": float(bound), "flux_norm_squared": derivative_part,
                "field_norm_squared": field_part, "mu": self.mu, "scale": self.scale,
                "deformation": "diagonal_dipole" if self.deformation is None else type(self.deformation).__name__,
                "quadrature_points": len(r), "angle_points_per_interval": self.weights.count,
                "scope": "Conditional analytic bound evaluated on a numerical background; not an enclosure."}

    def pairing(self, field, order=24):
        """field(r) returns two fields followed by their two radial derivatives."""
        r, q, R, j = self.quadrature(order)
        c = dipole_coefficients(self.weights.background, r)
        y = field(r)
        value = q@(j*y[0]-R*y[2]-2*self.weights.H(r)*np.sum(c["a"]*y[:2].T, axis=1))
        return float(value)
