"""Forced static even multipoles on a represented solitonic boson star.

The radiation source is a positive smooth mixture of trapped null orbits.
All boundary and solver outputs here are numerical, not interval certificates.
"""

from math import acosh, ceil, comb, isfinite, log

import numpy as np
from mpmath import mp
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicHermiteSpline
from scipy.linalg import expm

from finite_width_axial import OrbitSource
from trapped_axial import gauss_mesh


class SmoothOrbitSource:
    """Energy weight E*(Ec-E)^k on the inner trapped component, k=4."""
    def __init__(self, background, ceiling=.6, order=160):
        self.background = background
        self.orbit = OrbitSource(background, ceiling, order=order)
        self.a, self.b = self.orbit.a, self.orbit.b
        self.ceiling = ceiling
        self.energy = self.orbit.energy
        self.depth = self.energy-background.Us
        self.factor = 1.
        self.power, self.denominator = 4.5, 11.
        x, w = np.polynomial.legendre.leggauss(order)
        r = (self.a+self.b)/2+(self.b-self.a)*x/2
        rho, _, _, _ = self.fields(r)
        g = background.fields(r)
        norm = np.dot(w, 4*np.pi*r*r*g["alpha"]/np.sqrt(g["f"])*rho)*(self.b-self.a)/2
        self.factor = 1/norm
        self.knots = np.array([self.a, self.b])

    def fields(self, radius):
        r = np.atleast_1d(radius)
        if np.any(r <= 0):
            raise ValueError("Evaluate the compact radiation source at positive radius.")
        g = self.background.fields(r)
        gap = np.maximum(self.energy-g["U"], 0)
        shape = np.where((r > self.a) & (r < self.b), gap/self.depth, 0.)
        base = self.factor*shape**self.power/(4*np.pi*r*r*g["alpha"]**2)
        pressure = base*gap/self.denominator
        density = base*g["U"]+pressure
        D = self.factor*shape**self.power/(4*r*r)
        up = g["U"]*(-2*self.background.sol(r, 1)[3]-2/r)
        Dp = -2*D/r-self.factor*self.power*shape**(self.power-1)*up/(4*r*r*self.depth)
        return density, pressure, D, Dp

    def diagnostics(self, order=240):
        r, weights = gauss_mesh(self.knots, order)
        rho, pr, D, Dp = self.fields(r)
        g = self.background.fields(r)
        np_interp = -self.background.sol(r, 1)[3]
        # Analytic differentiation of the closed energy-mixture integral.
        prp = pr*(-2/r-2*np_interp)+(rho-pr)*(1/r-np_interp)
        residual = prp+(rho+pr)*g["nup"]+(3*pr-rho)/r
        energy = np.dot(weights, 4*np.pi*r*r*g["alpha"]/np.sqrt(g["f"])*rho)
        step = 2e-5
        interior = np.linspace(self.a+.01*(self.b-self.a), self.b-.01*(self.b-self.a), 101)
        derivative = (self.fields(interior+step)[2]-self.fields(interior-step)[2])/(2*step)
        error = np.max(abs(derivative-self.fields(interior)[3]))
        scaled_error = np.max(abs(derivative-self.fields(interior)[3])/(1+abs(self.fields(interior)[3])))
        rho_i, pr_i, _, _ = self.fields(interior)
        pressure_derivative = (self.fields(interior+step)[1]-self.fields(interior-step)[1])/(2*step)
        direct_conservation = (pressure_derivative+(rho_i+pr_i)*self.background.fields(interior)["nup"]
                               +(3*pr_i-rho_i)/interior)
        return {"energy_ceiling_fraction": self.ceiling, "support": [self.a, self.b],
                "energy": float(energy), "minimum_density": float(np.min(rho)),
                "maximum_pressure_fraction": float(np.max(pr/rho)),
                "conservation_residual": float(np.max(abs(residual))),
                "finite_difference_conservation_residual": float(np.max(abs(direct_conservation))),
                "anisotropic_derivative_difference": float(error),
                "scaled_anisotropic_derivative_difference": float(scaled_error)}

    def source(self, radius):
        r = np.atleast_1d(radius)
        if np.any(r < 0):
            raise ValueError("Use nonnegative radius.")
        values = np.zeros((5, len(r)))
        mask = r > 0
        rho, pr, _, _ = self.fields(r[mask])
        values[0, mask], values[1, mask], values[2, mask] = rho, pr, (rho-pr)/2
        return values


class OrbitBandSource(SmoothOrbitSource):
    """Positive compact energy band with weight E*[4t(1-t)]^4."""
    def __init__(self, background, band=(.8, .95), order=80):
        lower, upper = band
        if not 0 < lower < upper < 1:
            raise ValueError("Use an ordered energy band strictly inside the trapping well.")
        self.background, self.floor, self.ceiling = background, lower, upper
        self.orbit = OrbitSource(background, upper)
        inner = OrbitSource(background, lower)
        self.a, self.b, self.energy = self.orbit.a, self.orbit.b, self.orbit.energy
        self.depth = background.Uu-background.Us
        self.knots = np.array([self.a, inner.a, inner.b, self.b])
        self._nodes, self._weights = np.polynomial.legendre.leggauss(12)
        self.factor = 1.
        r, weights = gauss_mesh(self.knots, order)
        rho = self.fields(r)[0]
        g = background.fields(r)
        self.factor = 1/float(weights@(4*np.pi*r*r*g["alpha"]/np.sqrt(g["f"])*rho))

    def fields(self, radius):
        r = np.atleast_1d(radius)
        if np.any(r <= 0):
            raise ValueError("Use positive source radii.")
        g = self.background.fields(r)
        active = (r > self.a) & (r < self.b)
        result = np.zeros((4, len(r)))
        if not np.any(active):
            return result
        x = (g["U"][active]-self.background.Us)/self.depth
        left = np.sqrt(np.maximum(self.floor-x, 0))
        right = np.sqrt(np.maximum(self.ceiling-x, 0))
        half = (right-left)/2
        root = (right+left)[:, None]/2+half[:, None]*self._nodes
        fraction = (x[:, None]+root**2-self.floor)/(self.ceiling-self.floor)
        shape = 4*fraction*(1-fraction)
        I0 = 2*half*(shape**4@self._weights)
        I1 = 2*half*((shape**4*root**2)@self._weights)
        I0p = 2*half*(16*shape**3*(1-2*fraction)/(self.ceiling-self.floor)@self._weights)
        rr, U, alpha = r[active], g["U"][active], g["alpha"][active]
        prefactor = self.factor/(4*np.pi*rr*rr*alpha**2)
        pressure = prefactor*self.depth*I1
        density = prefactor*U*I0+pressure
        D = self.factor*I0/(4*rr*rr)
        up = U*(-2*self.background.sol(rr, 1)[3]-2/rr)
        Dp = -2*D/rr+self.factor*I0p*up/(4*rr*rr*self.depth)
        result[:, active] = density, pressure, D, Dp
        return result


def angular_parameters(degree):
    if not isinstance(degree, (int, np.integer)) or degree not in range(2, 257, 2):
        raise ValueError("This bounded numerical implementation supports even degrees 2 through 256.")
    degree = int(degree)
    eigenvalue = degree*(degree+1)
    funk = (-1)**(degree//2)*comb(degree, degree//2)/2**degree
    return eigenvalue, funk, -8*funk/(eigenvalue-2)


def coefficients(background, radius, degree=2):
    eigenvalue, _, _ = angular_parameters(degree)
    r = np.atleast_1d(radius)
    g = background.fields(r)
    A, Ap, f, np_, fp = [g[key] for key in ("A", "Ap", "f", "nup", "fp")]
    Om2, V, Vs = g["Omega"]**2, g["V"], g["Vs"]
    Vss = -8/background.scale**2+24*A*A/background.scale**4
    damping = 2/r+np_+fp/(2*f)
    F = 4*np.pi*(2*Om2*A*A-V)
    npp = F/f-np_**2-(2/r+fp/(2*f))*np_
    App = -damping*Ap+(Vs-Om2)*A/f
    cross = 4*np_*Ap+2*A*(2*Om2-Vs)/f
    vv = 4*np_**2+eigenvalue/(r*r*f)+8*np.pi*(V-4*Om2*A*A)/f
    uu = (eigenvalue/r**2+Vs+2*A*A*Vss-Om2)/f+16*np.pi*Ap*Ap
    return g, damping, F, npp, App, cross, vv, uu


def polar_matrix(background, radius, degree=2):
    r = np.atleast_1d(radius)
    g, damping, _, _, _, cross, vv, uu = coefficients(background, r, degree)
    central = background.central
    matrix = np.zeros((len(r), 5, 5))
    matrix[:, 0, 1] = 1
    matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2] = vv, -damping, 4*np.pi*central*cross
    matrix[:, 2, 3] = 1
    matrix[:, 3, 0], matrix[:, 3, 2], matrix[:, 3, 3] = cross/central, uu, -damping
    # The fifth state is w=c+v, which is O(r^(degree+2)) at the center.
    matrix[:, 4, 0], matrix[:, 4, 2] = -2*g["nup"], -8*np.pi*central*g["Ap"]
    return matrix


def polar_forcing(source, radius, degree=2):
    r, background = np.atleast_1d(radius), source.background
    rho, _, D, Dp = source.fields(r)
    _, funk, gain = angular_parameters(degree)
    D, Dp = gain*D, gain*Dp
    g, _, F, _, _, _, _, _ = coefficients(background, r, degree)
    np_, f, A, Ap = [g[key] for key in ("nup", "f", "A", "Ap")]
    result = np.zeros((5, len(r)))
    result[1] = (2*F/f-2*np_*(1/r+np_))*D+np_*Dp+8*np.pi*funk*rho/f
    result[3] = (Ap*(Dp-2*(1/r+np_)*D)+2*A*(g["Vs"]-g["Omega"]**2)*D/f)/background.central
    result[4] = (1/r+np_)*D
    return result


def polar_flux_forcing(source, radius, degree=2):
    """Forcing for (v, v'-nu'*D_l, u/A0, (u'-A'*D_l)/A0, w)."""
    r, background = np.atleast_1d(radius), source.background
    rho, _, D, _ = source.fields(r)
    _, funk, gain = angular_parameters(degree)
    D = gain*D
    g, _, F, _, _, _, _, _ = coefficients(background, r, degree)
    n, f, A, Ap = [g[key] for key in ("nup", "f", "A", "Ap")]
    result = np.zeros((5, len(r)))
    result[0] = n*D
    result[1] = (F/f-2*n*(1/r+n))*D+8*np.pi*funk*rho/f
    result[2] = Ap*D/background.central
    result[3] = (A*(g["Vs"]-g["Omega"]**2)/f-2*(1/r+n)*Ap)*D/background.central
    result[4] = (1/r+n)*D
    return result


def angular_constraint(background, radius, state, D=0., Dp=0., rho=0., degree=2):
    """Reconstruct c from the angular trace; D is already degree-dependent."""
    r = np.atleast_1d(radius)
    v, vp, u, up = state[:4]
    u, up = background.central*u, background.central*up
    eigenvalue, funk, _ = angular_parameters(degree)
    g, damping, F, npp, App, cross, vv, _ = coefficients(background, r, degree)
    f, np_, fp, A, Ap, V, Vs = [g[key] for key in ("f", "nup", "fp", "A", "Ap", "V", "Vs")]
    vpp = (-damping*vp+vv*v+4*np.pi*cross*u+(2*F/f-2*np_*(1/r+np_))*D
           +np_*Dp+8*np.pi*funk*rho/f)
    cp = -vp-2*np_*v+(1/r+np_)*D-8*np.pi*Ap*u
    cpp = -vpp-2*npp*v-2*np_*vp+(-1/r**2+npp)*D+(1/r+np_)*Dp-8*np.pi*(App*u+Ap*up)
    return (f*r*r*cpp+f*r*(4+r*np_+r*fp/(2*f))*cp-r*f*(Dp-2*vp)
            -2*(1-4*np.pi*r*r*V)*(D-v)-(eigenvalue-1)*D+8*np.pi*r*r*Vs*A*u)/(eigenvalue-2)


def vacuum_metric_slope(mass, radius, degree=2):
    eigenvalue, _, _ = angular_parameters(degree)
    if not isfinite(mass) or not isfinite(radius) or mass <= 0 or radius <= 2*mass:
        raise ValueError("The vacuum matching radius must exceed twice the positive mass.")
    # Forward Q_l recurrence loses about 2*l*acosh(x)/log(10) digits.
    digits = max(60, ceil(2*(degree+1)*acosh(radius/mass-1)/log(10))+40)
    with mp.workdps(digits):
        M, r = mp.mpf(float(mass)), mp.mpf(float(radius))
        x = r/M-1
        previous = mp.log((x+1)/(x-1))/2
        current = x*previous-1
        for ell in range(1, degree):
            previous, current = current, ((2*ell+1)*x*current-ell*previous)/(ell+1)
        derivative = degree*(x*current-previous)/(x*x-1)
        q = eigenvalue*current-2*x*derivative
        qp = (eigenvalue-2)*derivative-2*x*q/(x*x-1)
        return float(qp/(M*q))


class CompactSourcePrimitive:
    """Quadrature of the three compact terms separated by the derivative shift."""
    def __init__(self, source, degree, knots, order=8, max_points=16000):
        self.source, self.degree = source, degree
        if (isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer))
                or not 4 <= order <= 16 or isinstance(max_points, (bool, np.bool_))
                or not isinstance(max_points, (int, np.integer)) or not 4000 <= max_points <= 16000):
            raise ValueError("Use order 4..16 and a compact quadrature budget of 4000..16000.")
        self.max_points, self.maximum_points_used = max_points, 0
        self.knots = np.unique(np.r_[source.knots, knots[(knots > source.a) & (knots < source.b)]])
        self.nodes, self.weights = np.polynomial.legendre.leggauss(order)
        left, right = self.knots[:-1], self.knots[1:]
        integrals = self.integrate(left, right)
        self.prefix = np.column_stack((np.zeros(3), np.cumsum(integrals, axis=1)))

    def integrand(self, radius, derivative=0):
        r, bg = np.atleast_1d(radius), self.source.background
        g, _, _, npp, App, _, _, _ = coefficients(bg, r, self.degree)
        _, _, gain = angular_parameters(self.degree)
        _, _, D, Dp = self.source.fields(r)
        D, Dp = gain*D, gain*Dp
        if derivative == 0:
            return np.array([g["nup"]*D, g["Ap"]*D/bg.central, (1/r+g["nup"])*D])
        if derivative == 1:
            return np.array([npp*D+g["nup"]*Dp, (App*D+g["Ap"]*Dp)/bg.central,
                             (-1/r**2+npp)*D+(1/r+g["nup"])*Dp])
        raise ValueError("Use derivative zero or one.")

    def integrate(self, left, right):
        count = len(left)*len(self.nodes)
        if count > self.max_points:
            raise ValueError(f"The compact source primitive exceeds {self.max_points} quadrature points.")
        self.maximum_points_used = max(self.maximum_points_used, count)
        half = (right-left)/2
        r = (left+right)[:, None]/2+half[:, None]*self.nodes
        values = self.integrand(r.ravel()).reshape(3, len(left), len(self.nodes))
        return np.einsum("ijk,k->ij", values, self.weights)*half

    def __call__(self, radius):
        r = np.atleast_1d(radius)
        values = np.zeros((3, len(r)))
        values[:, r >= self.source.b] = self.prefix[:, -1, None]
        active = (r > self.source.a) & (r < self.source.b)
        if np.any(active):
            index = np.searchsorted(self.knots, r[active], side="right")-1
            values[:, active] = self.prefix[:, index]+self.integrate(self.knots[index], r[active])
        return values


class PolarResponse:
    def __init__(self, source, tolerance=1e-5, inner=.002, outer=None, max_nodes=4000, degree=2,
                 source_derivative_shift=False, source_integral_order=8):
        self.source, self.background = source, source.background
        if not isinstance(source_derivative_shift, bool):
            raise ValueError("The source derivative shift must be boolean.")
        self.source_derivative_shift = source_derivative_shift
        self.eigenvalue, self.funk, self.gain = angular_parameters(degree)
        self.degree = degree
        bg = self.background
        self.inner, self.outer = inner, bg.radius if outer is None else outer
        if not 0 < inner < source.a < source.b < self.outer <= bg.radius:
            raise ValueError("The radial domain must enclose the whole source.")
        if not 0 < tolerance < 1 or not 1200 <= max_nodes <= 4000:
            raise ValueError("Invalid tolerance or radial node budget.")
        self.metric_slope = vacuum_metric_slope(bg.mass, self.outer, degree)
        kappa = np.sqrt(1-bg.omega**2)
        beta = bg.mass*(2*bg.omega**2-1)/kappa
        self.scalar_slope = -kappa+(beta-1)/self.outer
        mesh = np.unique(np.r_[np.linspace(inner, self.outer, 601), np.geomspace(inner, max(inner, min(.1, self.outer)), 31), source.knots,
                               np.linspace(source.a, source.b, 181),
                               np.linspace(max(inner, source.a-1), min(source.b+2, self.outer), 161)])
        self.solution = solve_bvp(self.equation, self.boundary, mesh, np.zeros((5, len(mesh))),
                                  fun_jac=self.jacobian, tol=tolerance, max_nodes=max_nodes)
        if not self.solution.success:
            raise RuntimeError(f"Polar response stopped within {max_nodes} radial nodes: {self.solution.message}")
        self.tolerance = tolerance
        self._integrated_states = self.solution.sol.antiderivative()
        r = self.solution.x
        y = self.unshifted_state(r)
        if self.source_derivative_shift:
            self._source_primitive = CompactSourcePrimitive(source, degree, r, order=source_integral_order)
            primitive = self._source_primitive(r)
            initial = self.solution.sol(self.inner)
            integral = self._integrated_states(r)-self._integrated_states(self.inner)[:, None]
            y[0] = initial[0]+integral[1]+primitive[0]
            y[2] = initial[2]+integral[3]+primitive[1]
        g, _, _, npp, App, _, _, _ = coefficients(bg, r, degree)
        _, _, D, Dp = self.harmonic_source(r)
        v, vp, u, up = y[:4]
        u, up = bg.central*u, bg.central*up
        wp = -2*g["nup"]*v+(1/r+g["nup"])*D-8*np.pi*g["Ap"]*u
        wpp = (-2*npp*v-2*g["nup"]*vp+(-1/r**2+npp)*D+(1/r+g["nup"])*Dp
               -8*np.pi*(App*u+g["Ap"]*up))
        if self.source_derivative_shift:
            wp -= self._source_primitive.integrand(r)[2]
            wpp -= self._source_primitive.integrand(r, 1)[2]
        # Integrate the represented first derivatives to obtain a C2 metric.
        self._wprime = CubicHermiteSpline(r, wp, wpp)
        self._wintegral = self._wprime.antiderivative()

    def equation(self, r, y):
        forcing = polar_flux_forcing if self.source_derivative_shift else polar_forcing
        return np.einsum("nij,jn->in", polar_matrix(self.background, r, self.degree), y)+forcing(self.source, r, self.degree)

    def unshifted_state(self, radius, derivative=0):
        if derivative not in (0, 1):
            raise ValueError("Only state values and first derivatives are supported.")
        r = np.atleast_1d(radius)
        state = self.solution.sol(r, derivative)
        if self.source_derivative_shift:
            g, _, _, npp, App, _, _, _ = coefficients(self.background, r, self.degree)
            _, _, D, Dp = self.harmonic_source(r)
            if derivative == 0:
                state[1] += g["nup"]*D
                state[3] += g["Ap"]*D/self.background.central
            else:
                state[1] += npp*D+g["nup"]*Dp
                state[3] += (App*D+g["Ap"]*Dp)/self.background.central
        return state

    def jacobian(self, r, y):
        return np.moveaxis(polar_matrix(self.background, r, self.degree), 0, -1)

    def harmonic_source(self, r):
        rho, pr, D, Dp = self.source.fields(r)
        return rho, pr, self.gain*D, self.gain*Dp

    def boundary(self, left, right):
        g = self.background.fields(self.inner)
        central_w = -self.inner/(self.degree+2)*(2*g["nup"]*left[0]+8*np.pi*g["Ap"]*self.background.central*left[2])
        return np.array([left[1]-self.degree*left[0]/self.inner, left[3]-self.degree*left[2]/self.inner,
                         right[1]-self.metric_slope*right[0], right[3]-self.scalar_slope*right[2], left[4]-central_w])

    def optical(self, radius):
        r = np.atleast_1d(radius)
        if np.any(r < self.inner) or np.any(r > self.outer):
            raise ValueError("Polar evaluation exceeds its finite radial domain.")
        v, _, _, _, _, _, w, _, _ = self.physical_jet(r)
        _, _, D, _ = self.harmonic_source(r)
        return self.funk*(2*v-w), self.funk*(D-w)

    def physical_jet(self, radius):
        r = np.atleast_1d(radius)
        y, dy = self.unshifted_state(r), self.unshifted_state(r, 1)
        initial = self.solution.sol(self.inner)
        integral = self._integrated_states(r)-self._integrated_states(self.inner)[:, None]
        if self.source_derivative_shift:
            primitive = self._source_primitive(r)
            v = initial[0]+integral[1]+primitive[0]
            u = self.background.central*(initial[2]+integral[3]+primitive[1])
        else:
            v = initial[0]+integral[1]
            u = self.background.central*(initial[2]+integral[3])
        vp, vpp = y[1], dy[1]
        up, upp = self.background.central*y[3], self.background.central*dy[3]
        w = initial[4]+self._wintegral(r)-self._wintegral(self.inner)
        wp, wpp = self._wprime(r), self._wprime(r, 1)
        if self.source_derivative_shift:
            w += primitive[2]
            wp += self._source_primitive.integrand(r)[2]
            wpp += self._source_primitive.integrand(r, 1)[2]
        return v, vp, vpp, u, up, upp, w, wp, wpp

    def action_response(self, fraction, order=160):
        probe = OrbitSource(self.background, fraction, order=order)
        r, weights, gap = probe.raw_quadrature(order)
        weights /= np.sum(weights)
        v, k = self.optical(r)
        vstar = float(weights@(v-k*gap/probe.energy))
        r, weights, gap = probe.raw_quadrature(order, barrier=True)
        v, k = self.optical(r)
        distance = float(weights@(k*gap))
        frequency = float(weights@(probe.energy*(v-vstar)))
        result = {"fraction": fraction, "degree": self.degree, "D_degree": distance+frequency,
                  "distance_term": distance, "frequency_term": frequency, "vstar": vstar}
        if self.degree == 2:
            result.update(D2=distance+frequency, vstar2=vstar)
        return result

    def residuals(self):
        knots = self.solution.x
        r = knots[:-1]+.37*np.diff(knots)
        y, dy = [self.unshifted_state(r, derivative) for derivative in (0, 1)]
        v, vp, vpp, u, up, upp, w, wp, wpp = self.physical_jet(r)
        c, cp, cpp = w-v, wp-vp, wpp-vpp
        rho, pr, D, Dp = self.harmonic_source(r)
        b, bp = D-v, Dp-vp
        g, damping, F, _, _, _, _, _ = coefficients(self.background, r, self.degree)
        f, fp, np_, A, Ap, Vs, V = [g[key] for key in ("f", "fp", "nup", "A", "Ap", "Vs", "V")]
        Om2 = g["Omega"]**2
        Vss = -8/self.background.scale**2+24*A*A/self.background.scale**4
        R = 1-f-r*f*np_-r*fp/2
        geometry = np.array([
            f*(vpp+(2*np_+2/r+fp/(2*f))*vp+np_*(2*cp-bp))+2*(v-b)*F-self.eigenvalue*v/r**2,
            -vpp-2*cpp+np_*(bp-2*vp)+2*bp/r-(4/r+fp/f)*cp-fp*vp/(2*f)+self.eigenvalue*b/(r*r*f),
            -vp-cp+(v+b)/r+np_*(b-v),
            (2*(R-1)*c+2*(1-R)*b+self.eigenvalue*c+self.eigenvalue/2*(v+b)-f*r*r*cpp
             -f*r*(4+r*np_+r*fp/(2*f))*cp+r*f*(bp-vp))/(r*r),
            f*(upp+damping*up)+(Om2-Vs-2*A*A*Vss-self.eigenvalue/r**2)*u
            +f*(vp-bp+2*cp)*Ap-2*b*(Vs-Om2)*A-2*v*Om2*A,
        ])
        matter = np.array([
            8*np.pi*((2*Om2-Vs)*A*u-V*v)+8*np.pi*self.funk*rho,
            8*np.pi*(2*Ap*up+(V*b+Vs*A*u)/f)+8*np.pi*self.funk*pr/f,
            8*np.pi*Ap*u,
            8*np.pi*(V*c+Vs*A*u)+4*np.pi*self.funk*(rho-pr),
            np.zeros_like(r),
        ])
        names = ("tt", "rr", "rtheta", "angular_trace", "scalar")
        scaled = np.max(abs(geometry-matter)/(1+abs(geometry)+abs(matter)), axis=1)
        physical = np.array([v, vp, u/self.background.central, up/self.background.central, w])
        reconstructed = angular_constraint(self.background, r, physical, D, Dp, rho, self.degree)
        v, vp, _, u, up, _, w, _, _ = self.physical_jet([self.inner, self.outer])
        endpoints = np.array([v, vp, u/self.background.central, up/self.background.central, w])
        return {"radial_points": len(knots), "verification_points": len(r),
                "max_rms_residual": float(np.max(self.solution.rms_residuals)),
                "kinematic_defect": float(np.max(abs(dy[[0, 2]]-y[[1, 3]]))),
                "field_reconstruction_difference": float(np.max(abs(physical-y))),
                "independent_equation_residuals": dict(zip(names, map(float, scaled))),
                "angular_constraint_difference": float(np.max(abs(reconstructed-c))),
                "inner_c_plus_v": float(self.solution.sol(self.inner)[4]),
                "reconstructed_boundary_residual": float(np.max(abs(self.boundary(endpoints[:, 0], endpoints[:, 1])))),
                "boundary_residual": float(np.max(abs(self.boundary(self.solution.sol(self.inner), self.solution.sol(self.outer)))))}


def monopole_action(response, fraction, order=160):
    bg = response.background
    probe = OrbitSource(bg, fraction, order=order)
    r, weights, gap = probe.raw_quadrature(order)
    weights /= np.sum(weights)
    x, v = response.metric(r)
    k = x/(r*bg.fields(r)["f"])
    vstar = float(weights@(v-k*gap/probe.energy))
    r, weights, gap = probe.raw_quadrature(order, barrier=True)
    x, v = response.metric(r)
    k = x/(r*bg.fields(r)["f"])
    return float(weights@(k*gap+probe.energy*(v-vstar)))


def matching_diagnostics(background, resolution=1, inner=.002, outer=None):
    if resolution not in (1, 2):
        raise ValueError("Use one of the two bounded matching resolutions.")
    R = background.radius if outer is None else outer
    middle = background.rs
    left_mesh = np.unique(np.r_[np.geomspace(inner, .2, 40*resolution+1),
                                np.linspace(.2, middle, 400*resolution+1)])
    right_mesh = np.linspace(R, middle, 600*resolution+1)
    if len(left_mesh)+len(right_mesh) > 4000:
        raise RuntimeError("The polar matching budget exceeds 4000 radial nodes.")
    left = np.array([[inner**2, 0], [2*inner, 0], [0, inner**2], [0, 2*inner]])
    kappa = np.sqrt(1-background.omega**2)
    beta = background.mass*(2*background.omega**2-1)/kappa
    right = np.array([[1., 0], [vacuum_metric_slope(background.mass, R), 0],
                      [0, 1], [0, -kappa+(beta-1)/R]])
    def transport(mesh, basis):
        Q = np.linalg.qr(basis, mode="reduced")[0]
        steps = np.diff(mesh)
        midpoint = (mesh[:-1]+mesh[1:])/2
        matrices1 = polar_matrix(background, midpoint-steps/(2*np.sqrt(3)))[:, :4, :4]
        matrices2 = polar_matrix(background, midpoint+steps/(2*np.sqrt(3)))[:, :4, :4]
        for h, first, second in zip(steps, matrices1, matrices2):
            exponent = h*(first+second)/2-np.sqrt(3)*h*h*(first@second-second@first)/12
            Q = np.linalg.qr(expm(exponent)@Q, mode="reduced")[0]
        return Q
    L, R = transport(left_mesh, left), transport(right_mesh, right)
    singular = np.linalg.svd(np.column_stack((L, R)), compute_uv=False)
    return {"resolution": resolution, "radial_points": len(left_mesh)+len(right_mesh),
            "singular_values": singular.tolist(), "minimum_singular_value": float(singular[-1])}
