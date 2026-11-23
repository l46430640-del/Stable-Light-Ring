"""Scalar channel transverse to the exact dipole displacement.

The coefficient formulas are exact on an exact background. Shooting on
the saved floating background is exploratory and is not a certificate.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline
from pathlib import Path
import hashlib
import json

from polar_static import coefficients


def transverse_coefficients(background, radius):
    r = np.atleast_1d(np.asarray(radius, dtype=float))
    if r.ndim != 1 or np.any(~np.isfinite(r)) or np.any(r <= 0) or np.any(r > background.radius):
        raise ValueError("Use positive radii inside the represented background.")
    g, d, _, np_, App, C, Av, Au = coefficients(background, r)
    a = np.column_stack((g["nup"], np.sqrt(4*np.pi)*g["Ap"]))
    ap = np.column_stack((np_, np.sqrt(4*np.pi)*App))
    norm2 = np.sum(a*a, axis=1)
    if np.any(norm2 <= 0):
        raise ValueError("The displacement vector must not vanish.")
    e = a/np.sqrt(norm2)[:, None]
    normal = np.column_stack((-e[:, 1], e[:, 0]))
    turn = (a[:, 0]*ap[:, 1]-a[:, 1]*ap[:, 0])/norm2
    gamma = np.sum(a*ap, axis=1)/norm2-g["nup"]+g["fp"]/(2*g["f"])
    W = 1/(r*r*g["f"])
    Q1 = np.zeros((len(r), 2, 2))
    Q1[:, 0, 0], Q1[:, 1, 1] = Av-4*W, Au-4*W
    Q1[:, 0, 1] = Q1[:, 1, 0] = np.sqrt(4*np.pi)*C
    potential = np.einsum("ni,nij,nj->n", normal, Q1, normal)-3*turn*turn
    return {"d": d, "W": W, "potential": potential, "turn": turn, "gamma": gamma, "e": e,
            "normal": normal, "a": a, "Q1": Q1, "P": r*r*g["p"]}


class TransverseDeformation:
    """Floating evaluation of the certified deformation on its reference star."""
    breakpoints = (15., 20.)

    def __init__(self, background):
        root = Path(__file__).parent
        background_path = root/"data/s08a06/background.npz"
        background_hash = hashlib.sha256(background_path.read_bytes()).hexdigest()
        with np.load(background_path) as data:
            if (background.central != .06 or background.scale != .08
                    or not np.array_equal(background.sol.x, data["r"])
                    or not np.array_equal(background.sol.c, data["coefficients"])):
                raise ValueError("This deformation is certified only for the original reference star.")
        path = root/"data/solitonic-screen/polar-transverse-deformation.npz"
        certificate = json.loads((root/"data/solitonic-screen/polar-transverse.json").read_text())
        if (certificate["status"] != "all_polar_degrees_coercivity_verified"
                or certificate["input_sha256"]["background.npz"] != background_hash
                or certificate["deformation_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError("The matrix deformation does not match its interval certificate.")
        with np.load(path) as data:
            if str(data["background_sha256"]) != background_hash:
                raise ValueError("The deformation belongs to a different background.")
            self.spline = CubicHermiteSpline(data["mesh"], data["values"], data["slopes"])
        self.background = background

    def __call__(self, radius):
        r = np.atleast_1d(np.asarray(radius, dtype=float))
        c = transverse_coefficients(self.background, r)
        s = np.zeros_like(r)
        core, wall = r <= 15, (r > 15) & (r <= 20)
        s[core] = 11*r[core]/(4*(1+r[core]))
        s[wall] = self.spline(r[wall])
        e, normal = c["e"], c["normal"]
        tensor = lambda x, y: np.einsum("ni,nj->nij", x, y)
        T = c["gamma"][:, None, None]*tensor(e, e)+s[:, None, None]*tensor(normal, normal)
        T += c["turn"][:, None, None]*(tensor(e, normal)+tensor(normal, e))
        return T


def shoot(background, tolerance=1e-9, shift=0., start=.002):
    if not np.isfinite(tolerance) or not 1e-11 <= tolerance <= 1e-5:
        raise ValueError("Use a shooting tolerance from 1e-11 through 1e-5.")
    if not np.isfinite(shift) or not 0 <= shift <= 4:
        raise ValueError("Use a transverse centrifugal shift between zero and four.")
    if not .001 <= start <= .01:
        raise ValueError("Use a small positive central cutoff.")
    power = (-1+np.sqrt(9+4*shift))/2
    c = transverse_coefficients(background, [start])
    remainder = c["potential"][0]+shift*c["W"][0]-power*(power-1)/start**2-power*c["d"][0]/start
    zprime = start*remainder/(2*power+3)
    evaluations = 0
    def equation(r, z):
        nonlocal evaluations
        evaluations += 1
        if evaluations > 48000:
            raise RuntimeError("The bounded transverse shooting evaluation budget was exhausted.")
        c = transverse_coefficients(background, [r])
        d = c["d"][0]
        potential = c["potential"][0]+shift*c["W"][0]
        return [z[1], -(d+2*power/r)*z[1]+(potential-power*(power-1)/r**2-power*d/r)*z[0]]
    def zero(r, z):
        return z[0]
    zero.terminal, zero.direction = True, -1
    solution = solve_ivp(equation, (start, background.radius), [1., zprime], method="DOP853",
                         rtol=tolerance, atol=tolerance/100, max_step=.05, dense_output=True, events=zero)
    if not solution.success or len(solution.t) > 4000:
        raise RuntimeError("The transverse shooting calculation did not complete within its budget.")
    final = solution.y[:, -1]
    return solution, {"scope": "Floating-background scalar shooting; no certified absence of zeros.",
        "shift": shift, "tolerance": tolerance, "start": start, "central_power": power,
        "radial_nodes": len(solution.t), "evaluations": evaluations,
        "first_zero": float(solution.t_events[0][0]) if len(solution.t_events[0]) else None,
        "final_radius": float(solution.t[-1]), "final_normalized_field": float(final[0]),
        "final_logarithmic_slope": float(final[1]/final[0]+power/solution.t[-1]) if final[0] != 0 else None}


def deformation_proposal(background, margin=.05, tolerance=1e-9):
    """Propose a scalar logarithmic derivative on the finite scalar wall."""
    if not np.isfinite(margin) or not 0 < margin <= .2:
        raise ValueError("Use a positive deformation margin at most .2.")
    if not np.isfinite(tolerance) or not 1e-11 <= tolerance <= 1e-6:
        raise ValueError("Use a proposal tolerance between 1e-11 and 1e-6.")
    evaluations = 0
    def equation(r, s):
        nonlocal evaluations
        evaluations += 1
        if evaluations > 12000:
            raise RuntimeError("The bounded deformation proposal budget was exhausted.")
        c = transverse_coefficients(background, [r])
        return c["potential"]-c["d"]*s-s*s-margin
    def lower_limit(r, s):
        return s[0]+10
    lower_limit.terminal, lower_limit.direction = True, -1
    solution = solve_ivp(equation, (15., 20.), [165/64], rtol=tolerance, atol=tolerance/100,
                         method="DOP853", max_step=.02, dense_output=True, events=lower_limit)
    if not solution.success or solution.t[-1] != 20. or len(solution.t) > 1000:
        raise RuntimeError(f"The scalar deformation proposal stopped at r={solution.t[-1]:.8g}, s={solution.y[0,-1]:.8g}.")
    return solution, {"margin": margin, "radial_nodes": len(solution.t), "evaluations": evaluations,
                      "minimum_slope": float(min(solution.y[0])), "final_slope": float(solution.y[0, -1])}


if __name__ == "__main__":
    import json
    from trapped_axial import Background
    bg = Background()
    r = np.linspace(.002, bg.radius, 1601)
    c = transverse_coefficients(bg, r)
    i = int(np.argmin(c["potential"]))
    print(json.dumps({"sampled_potential_minimum": float(c["potential"][i]), "radius": float(r[i])}))
    print(json.dumps(shoot(bg)[1]))
