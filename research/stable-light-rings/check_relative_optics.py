"""Enclose optical potential ratios by integrating the exact logarithmic slope.

All radii and stored background polynomials retain their exact binary values.
The absolute potential scale cancels before the shallow-well division is taken.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
from math import comb
from pathlib import Path
from time import perf_counter

if not __debug__:
    raise SystemExit("Run optical enclosures without python -O.")

import numpy as np
from mpmath import iv

from check_asymptotic_exterior import lower
from check_background_defect import field_equations
from check_barrier_certificate import Jet, exact, enclosure
from check_finite_static_inverse import upper
from check_physical_barrier import STABLE, UNSTABLE, physical_box, optical_quantities, optical_H_derivative
from check_polar_band_source import endpoints
from check_polar_coercivity import check_inputs, physical_cell
from trapped_axial import Background

ROOT = Path(__file__).parent
DEFAULT = ROOT/"data/solitonic-screen/relative-optics.json"
DOMAIN = (Fraction("15.62"), Fraction("16.55"))
ORDER = 4


def verified_background(bg):
    hashes = check_inputs()
    with np.load(ROOT/"data/s08a06/background.npz") as saved:
        assert bg.central == .06 and bg.scale == .08
        assert np.array_equal(bg.sol.x, saved["r"])
        assert np.array_equal(bg.sol.c, saved["coefficients"])
    record = json.loads((ROOT/"data/s08a06/global-background.json").read_text())
    assert [v["radius_interval"] for v in record["light_rings"]] == [list(STABLE), list(UNSTABLE)]
    return hashes


def slope_jet(radius, state):
    r = Jet(0, [radius, iv.mpf(1)]+[iv.mpf(0)]*(ORDER-1))
    y = [Jet(v, order=ORDER) for v in state]
    for k in range(ORDER):
        F = field_equations(r, y, iv.mpf(".06"), iv.mpf(".08"), iv.pi)
        for j in range(4):
            y[j].c[k+1] = F[j].c[k]/(k+1)
    F = field_equations(r, y, iv.mpf(".06"), iv.mpf(".08"), iv.pi)
    return -2*F[3]-2/r


def refined_ring(bg, bracket):
    a, b = map(Fraction, bracket)
    initial = (a, b)
    for _ in range(4):
        m = (a+b)/2
        H = optical_quantities(exact(m), physical_box(bg, m, m), iv.mpf(".06"), iv.mpf(".08"))[0]
        derivative = optical_H_derivative(enclosure(a, b), physical_box(bg, a, b), iv.mpf(".06"), iv.mpf(".08"))
        assert derivative.a > 0 or derivative.b < 0
        narrowed = exact(m)-H/derivative
        a, b = max(a, Fraction(lower(narrowed))), min(b, Fraction(upper(narrowed)))
        assert a < b
    assert initial[0] <= a < b <= initial[1]
    return a, b


class RelativePotential:
    def __init__(self, bg):
        self.bg, self.hashes = bg, verified_background(bg)
        self.anchor = Fraction(float(bg.ru))
        stable, unstable = refined_ring(bg, STABLE), refined_ring(bg, UNSTABLE)
        self.ring_brackets = [stable, unstable]
        self.knots = [Fraction(float(x)) for x in bg.sol.x]
        self.nodes = sorted({*DOMAIN, self.anchor, Fraction(float(bg.rs)), *stable, *unstable}
                            | {x for x in self.knots if DOMAIN[0] < x < DOMAIN[1]})
        assert len(self.nodes) <= 1200
        self.center, self.ranged, self.integrals = [], [], []
        start = perf_counter()
        for a, b in zip(self.nodes[:-1], self.nodes[1:]):
            if perf_counter()-start > 120:
                raise RuntimeError("Relative optical integration exceeded 120 seconds.")
            m = (a+b)/2
            piece = bisect_right(self.knots, m)-1
            center = slope_jet(exact(m), physical_cell(bg, piece, m, m)).c
            ranged = slope_jet(enclosure(a, b), physical_cell(bg, piece, a, b)).c
            offset = enclosure(-(b-a)/2, (b-a)/2)
            for k in range(ORDER):
                taylor = sum((comb(n, k)*center[n]*offset**(n-k) for n in range(k, ORDER)), iv.mpf(0))
                taylor += comb(ORDER, k)*ranged[ORDER]*offset**(ORDER-k)
                ranged[k] = enclosure(max(taylor.a, ranged[k].a), min(taylor.b, ranged[k].b))
            self.center.append(center)
            self.ranged.append(ranged)
            self.integrals.append(self.integral(len(self.center)-1, a, b))
        index = self.nodes.index(self.anchor)
        self.log_nodes = [iv.mpf(0) for _ in self.nodes]
        for j in range(index, len(self.nodes)-1):
            self.log_nodes[j+1] = self.log_nodes[j]+self.integrals[j]
        for j in range(index-1, -1, -1):
            self.log_nodes[j] = self.log_nodes[j+1]-self.integrals[j]
        self.log_rings = [self.ring_log(Fraction(float(reference)), bracket)
                          for reference, bracket in zip((bg.rs, bg.ru), self.ring_brackets)]
        self.depth_ratio = 1-iv.exp(self.log_rings[0]-self.log_rings[1])
        assert self.depth_ratio.a > 0

    def integral(self, index, left, right):
        a, b = self.nodes[index:index+2]
        assert a <= left <= right <= b
        m = (a+b)/2
        u, v = exact(left-m), exact(right-m)
        value = sum((self.center[index][k]*(v**(k+1)-u**(k+1))/(k+1)
                     for k in range(ORDER)), iv.mpf(0))
        # ORDER is even, so the remainder integral is positive also across m.
        error = abs(self.ranged[index][ORDER])*(v**(ORDER+1)-u**(ORDER+1))/(ORDER+1)
        return value+enclosure(-error.b, error.b)

    def log_point(self, radius):
        assert DOMAIN[0] <= radius <= DOMAIN[1]
        i = min(bisect_right(self.nodes, radius)-1, len(self.nodes)-2)
        a, b = self.nodes[i:i+2]
        if b <= self.anchor:
            return self.log_nodes[i+1]-self.integral(i, radius, b)
        return self.log_nodes[i]+self.integral(i, a, radius)

    def log_ratio(self, left, right):
        assert DOMAIN[0] <= left <= right <= DOMAIN[1]
        m = (left+right)/2
        i = min(bisect_right(self.nodes, m)-1, len(self.nodes)-2)
        assert self.nodes[i] <= left <= right <= self.nodes[i+1]
        return self.log_point(m)+enclosure(-(right-left)/2, (right-left)/2)*self.ranged[i][0]

    def ring_log(self, reference, bracket):
        a, b = bracket
        extent = (min(a, reference), max(b, reference))
        h0 = slope_jet(exact(reference), physical_box(self.bg, reference, reference)).c[0]
        h1 = slope_jet(enclosure(*extent), physical_box(self.bg, *extent)).c[1]
        delta = enclosure(a-reference, b-reference)
        return self.log_point(reference)+h0*delta+h1*delta**2/2

    def fraction(self, left, right):
        relative = iv.exp(self.log_ratio(left, right)-self.log_rings[1])
        return 1-(1-relative)/self.depth_ratio, relative/self.depth_ratio

    def turning_bracket(self, left, right, fraction=Fraction(".9985")):
        a, b = map(Fraction, (left, right))
        def gap(r):
            return self.fraction(r, r)[0]-exact(fraction)
        ga, gb = gap(a), gap(b)
        assert ga.a > 0 and gb.b < 0 or ga.b < 0 and gb.a > 0
        increasing = ga.b < 0
        for _ in range(32):
            m = (a+b)/2
            g = gap(m)
            if g.a > 0:
                if increasing:
                    b = m
                else:
                    a = m
            elif g.b < 0:
                if increasing:
                    a = m
                else:
                    b = m
            else:
                # Refine each endpoint toward the unresolved center separately.
                for side in (0, 1):
                    x = (a+m)/2 if side == 0 else (m+b)/2
                    gx = gap(x)
                    if (gx.b < 0 if increasing else gx.a > 0) and side == 0:
                        a = x
                    if (gx.a > 0 if increasing else gx.b < 0) and side == 1:
                        b = x
        return {"bracket": [str(a), str(b)], "width_upper": upper(exact(b-a)),
                "endpoint_fraction_gaps": [endpoints(gap(a)), endpoints(gap(b))]}

    def cumulative_center_bound(self):
        left, right = Fraction("16.50"), Fraction("16.55")
        nodes = sorted({left, right} | {v for v in self.nodes if left < v < right})
        curvature, reciprocal = float("inf"), 0.
        for a, b in zip(nodes[:-1], nodes[1:]):
            i = bisect_right(self.nodes, (a+b)/2)-1
            h, hp = self.ranged[i][:2]
            Q = iv.exp(self.log_ratio(a, b))
            curvature = min(curvature, lower(-Q*(h*h+hp)))
            piece = bisect_right(self.knots, (a+b)/2)-1
            f = 1-2*physical_cell(self.bg, piece, a, b)[2]
            reciprocal = max(reciprocal, upper(1/(enclosure(a, b)*iv.sqrt(Q*f))))
        assert curvature > 0
        depth = iv.exp(self.log_rings[1])*self.depth_ratio
        H0 = iv.pi*iv.sqrt(2/exact(curvature))*exact(reciprocal)*depth
        return {"concavity_interval": [str(left), str(right)], "cells": len(nodes)-1,
                "minus_relative_potential_second_derivative_lower": curvature,
                "inverse_radial_optical_scale_upper": reciprocal,
                "absolute_cumulative_weight_at_center_upper": upper(H0)}

    def record(self):
        return {"status": "relative_optical_potential_enclosed", "input_sha256": self.hashes,
            "anchor": str(self.anchor), "order": ORDER,
            "nodes": list(map(str, self.nodes)),
            "center_slope_jets": [[endpoints(v) for v in row] for row in self.center],
            "range_slope_jets": [[endpoints(v) for v in row] for row in self.ranged],
            "log_nodes": [endpoints(v) for v in self.log_nodes],
            "ring_brackets": [list(map(str, bracket)) for bracket in self.ring_brackets],
            "log_ring_ratios": [endpoints(v) for v in self.log_rings],
            "relative_well_depth": endpoints(self.depth_ratio)}

    @classmethod
    def load(cls, bg, path=DEFAULT):
        record = json.loads(path.read_text())
        assert record["status"] == "relative_optical_potential_enclosed" and record["order"] == ORDER
        result = cls.__new__(cls)
        result.bg, result.hashes = bg, verified_background(bg)
        assert result.hashes == record["input_sha256"]
        result.record_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        result.nodes = list(map(Fraction, record["nodes"]))
        result.anchor = Fraction(record["anchor"])
        result.knots = [Fraction(float(v)) for v in bg.sol.x]
        assert result.nodes[0] == DOMAIN[0] and result.nodes[-1] == DOMAIN[1]
        assert all(a < b for a, b in zip(result.nodes[:-1], result.nodes[1:]))
        result.center = [[enclosure(*v) for v in row] for row in record["center_slope_jets"]]
        result.ranged = [[enclosure(*v) for v in row] for row in record["range_slope_jets"]]
        result.log_nodes = [enclosure(*v) for v in record["log_nodes"]]
        assert len(result.center) == len(result.ranged) == len(result.nodes)-1
        assert len(result.log_nodes) == len(result.nodes)
        assert all(len(row) == ORDER+1 for row in result.center+result.ranged)
        result.log_rings = [enclosure(*v) for v in record["log_ring_ratios"]]
        result.ring_brackets = [tuple(map(Fraction, row)) for row in record["ring_brackets"]]
        result.depth_ratio = 1-iv.exp(result.log_rings[0]-result.log_rings[1])
        assert result.depth_ratio.a > 0
        return result


def implementation_checks(potential):
    from polar_static import coefficients
    from polar_band_moments import band_moments, _raw_band_moments
    bg = potential.bg
    z0 = bg.sol(float(potential.anchor))[3]
    count, difference = 0, 0.
    for radius in np.linspace(float(DOMAIN[0]), float(DOMAIN[1]), 31):
        r = min(max(Fraction(float(radius)), DOMAIN[0]), DOMAIN[1])
        reference = -2*(bg.sol(float(r))[3]-z0)-2*np.log(float(r/potential.anchor))
        box = potential.log_point(r)
        assert box.a-iv.mpf("2e-15") <= reference <= box.b+iv.mpf("2e-15")
        count += 1
        g, _, _, np_, *_ = coefficients(bg, [float(r)])
        h = slope_jet(exact(r), [exact(float(v)) for v in bg.sol(float(r))])
        expected = (2*(g["nup"][0]-1/float(r)), 2*(np_[0]+1/float(r)**2))
        difference = max(difference, *(abs(float(h.c[k].mid)-v) for k, v in enumerate(expected)))
    assert difference < 1e-12
    derivative_checks = 0
    for r in (".9", ".9979", ".998", ".9981", ".9983673", ".9985", ".9989", ".999", "1"):
        x = iv.mpf(r)
        box = band_moments(x, derivative=True)[2]
        old = _raw_band_moments(x, derivative=True)[2]
        assert box.a <= old.b and old.a <= box.b
        derivative_checks += 1
    for a, b in ((".9979", ".9981"), (".9982", ".9985"), (".9989", ".9991")):
        box = enclosure(iv.mpf(a), iv.mpf(b))
        derivative = band_moments(box, derivative=True)[2]
        for x in (box.a, box.mid, box.b):
            old = _raw_band_moments(x, derivative=True)[2]
            assert derivative.a <= old.b and old.a <= derivative.b
            derivative_checks += 1
    return {"relative_log_reference_checks": count, "slope_and_curvature_formula_difference": difference,
            "independent_density_derivative_checks": derivative_checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, choices=(30, 50), default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args()
    iv.dps = args.digits
    if args.checks_only:
        result = implementation_checks(RelativePotential.load(Background(), args.output))
        print(json.dumps(result, indent=2))
        return
    started = perf_counter()
    report = {"status": "running", "digits": args.digits}
    try:
        bg = Background()
        potential = RelativePotential(bg)
        report.update(potential.record())
        report["probe_fraction"] = ".9985"
        report["turning_points"] = [potential.turning_bracket(*pair) for pair in
                                     (("15.623", "15.624"), ("16.514", "16.516"), ("16.539", "16.541"))]
        report["central_optical_weight"] = potential.cumulative_center_bound()
        report["sample_fraction_ranges"] = {r: endpoints(potential.fraction(Fraction(r), Fraction(r))[0])
                                            for r in ("15.6234", "15.8", "16", "16.515", "16.517")}
    except (RuntimeError, ValueError, AssertionError) as error:
        report.update(status="stopped", reason=str(error))
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        raise
    report["elapsed_seconds"] = perf_counter()-started
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in
        ("nodes", "center_slope_jets", "range_slope_jets", "log_nodes", "input_sha256")}, indent=2))


if __name__ == "__main__":
    main()
