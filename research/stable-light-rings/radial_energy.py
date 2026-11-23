"""Constraint-reduced ADM mass variations for radial fixed-charge data.

The background and branch tangent are numerical inputs. No spectral or
interval certification is performed here.
"""

import numpy as np
from scipy.integrate import DOP853

from boson_star import potential
from static_transversality import background_charge


def bounded_integrate(function, radius, size, tolerance=1e-9, max_points=4000, max_step=.05):
    if not 2 <= max_points <= 4000 or tolerance <= 0 or max_step <= 0:
        raise ValueError("Use a positive tolerance and at most 4000 radial points.")
    solver = DOP853(function, 0., np.zeros(size), radius, rtol=tolerance,
                    atol=tolerance*1e-3, max_step=max_step)
    points = 1
    while solver.status == "running":
        if points >= max_points:
            raise RuntimeError("Constraint integration reached its radial point limit.")
        solver.step()
        points += 1
    if solver.status != "finished":
        raise RuntimeError("Constraint integration failed.")
    return solver.y, points


class RadialEnergy:
    """Use the branch tangent with delta A0=A0, corrected to zero charge."""

    def __init__(self, background, tangent):
        self.background, self.tangent = background, tangent
        self.charge = background_charge(background)
        self.charge_tangent = float(tangent.charge_response)
        self.omega_tangent = float(background.omega*tangent.dlogomega)
        self.charge_fraction = self.charge_tangent/self.charge

    def coefficients(self, r):
        bg = self.background
        g = bg.fields(r)
        state = self.tangent.solution.sol(r)
        A, Ap = float(g["A"]), float(g["Ap"])
        B, Bp = bg.central*state[0], bg.central*state[1]
        relative_A = B/A
        relative_charge = 2*relative_A+state[3]+state[2]/g["f"]-self.charge_fraction
        V, Vs = potential(A, bg.scale)
        Vss = -8/bg.scale**2+24*A*A/bg.scale**4
        area = 2*np.pi*r*r
        H = area*g["Omega"]**2*A*A/g["f"]
        K = area*Ap*Ap+H
        K1 = 2*area*Ap*Bp+2*H*(relative_charge-relative_A)
        K2 = 2*area*Bp*Bp+2*H*(relative_charge**2-4*relative_charge*relative_A+3*relative_A**2)
        V1 = 2*area*A*Vs*B
        V2 = 2*area*(Vs+2*A*A*Vss)*B*B
        charge_density = 2*area*g["Omega"]*A*A/np.sqrt(g["f"])
        return K, H, area*V, K1, K2, V1, V2, charge_density*relative_charge, (A, Ap, Bp, relative_A, relative_charge)

    def variations(self, tolerance=1e-9, max_step=.05):
        def rhs(r, state):
            if r == 0:
                return np.zeros(6)
            K, H, V, K1, K2, V1, V2, charge_change, _ = self.coefficients(r)
            mass, first, second, scale_first, scale_second, _ = state
            f, a = 1-2*mass/r, 2*K/r
            return np.array([f*K+V, -a*first+f*K1+V1,
                             -a*second+f*K2+V2-4*K1*first/r,
                             -a*scale_first+2*f*H,
                             -a*scale_second+2*f*H-8*H*scale_first/r,
                             charge_change])
        state, points = bounded_integrate(rhs, self.background.radius, 6, tolerance, max_step=max_step)
        mass, first, second, scale_first, scale_second, charge_change = state
        base = -self.omega_tangent*self.charge_tangent
        predicted = base+self.charge_fraction**2*scale_second
        upper = base+self.background.omega*self.charge_tangent**2/self.charge
        return {"central_amplitude": float(self.background.central),
                "outer_radius": float(self.background.radius), "charge": self.charge,
                "omega": float(self.background.omega),
                "dN_dA0": self.charge_tangent/self.background.central,
                "domega_dA0": self.omega_tangent/self.background.central,
                "normalization": "delta A0=A0; charge density corrected by -(delta N/N) times itself",
                "direct_mass_second_variation": float(second),
                "projected_hessian": float(predicted), "analytic_hessian_upper_bound": float(upper),
                "projection_discrepancy": float(abs(second-predicted)),
                "mass_first_variation": float(first), "integrated_charge_variation": float(charge_change),
                "background_mass_reconstruction_error": float(abs(mass-self.background.mass)),
                "charge_scaling_first_variation": float(scale_first),
                "charge_scaling_second_variation": float(scale_second),
                "charge_scaling_first_law_error": float(abs(scale_first-self.background.omega*self.charge)),
                "charge_scaling_upper_gap": float(self.background.omega*self.charge-scale_second),
                "integration_tolerance": tolerance, "maximum_step": max_step, "radial_points": points}

    def finite_difference(self, amplitude, tolerance=1e-10):
        if amplitude <= 0:
            raise ValueError("Use a positive finite-difference amplitude.")
        def rhs(r, state):
            if r == 0:
                return np.zeros(3)
            K, H, V, _, _, _, _, _, data = self.coefficients(r)
            A, Ap, Bp, a, c = data
            f = 1-2*state[0]/r
            result = [f*K+V]
            for sign, difference in zip((1., -1.), state[1:]):
                step = sign*amplitude
                if 1+step*a <= 0 or 1+step*c <= 0:
                    raise ValueError("Finite perturbation left the positive amplitude/charge domain.")
                changeK = 2*np.pi*r*r*(2*step*Ap*Bp+step*step*Bp*Bp)
                changeK += H*step*(c-a)*(2+step*(c+a))/(1+step*a)**2
                log_ratio = np.log1p(step*a)
                changeV = A*A*np.expm1(2*log_ratio)
                changeV -= 4*A**4/self.background.scale**2*np.expm1(4*log_ratio)
                changeV += 4*A**6/self.background.scale**4*np.expm1(6*log_ratio)
                result.append(-2*(K+changeK)*difference/r+f*changeK+2*np.pi*r*r*changeV)
            return np.array(result)
        state, points = bounded_integrate(rhs, self.background.radius, 3, tolerance)
        return {"amplitude": amplitude, "mass_changes": state[1:].tolist(),
                "second_difference": float((state[1]+state[2])/amplitude**2), "radial_points": points}
