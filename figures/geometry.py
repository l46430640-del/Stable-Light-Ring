"""Orbital phase and normalized quadrupole pattern in the Letter figures."""
from itertools import pairwise
from math import atan2, dist, hypot, pi, remainder
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import sympy as sp
HERE = Path(__file__).resolve().parent
SVG = "{http://www.w3.org/2000/svg}"


def by_id(root, name):
    node = root.find(f".//*[@id='{name}']")
    assert node is not None, name
    return node


def vertices(node):
    return [tuple(map(float, pair.split(","))) for pair in re.findall(
        r"[-\d.]+,[-\d.]+", node.attrib["d"])]


def check_planes(root):
    markers = [n for n in by_id(root, "normal-azimuths").iter(SVG+"circle") if n.get("fill") == "black"]
    assert len(markers) == 5
    xy = [(float(n.get("cx")), float(n.get("cy"))) for n in markers]
    u, v = (sum(axis)/5 for axis in zip(*xy))
    phis = sorted(atan2(b-v, a-u) for a, b in xy)
    assert all(abs(b-a-2*pi/5) < 1e-12 for a, b in pairwise(phis + [phis[0]+2*pi]))


def check_pattern(root):
    t = sp.symbols("t", real=True)
    y = sp.legendre(2, sp.cos(t))
    eigenvalue = sp.trigsimp((sp.diff(y, t, 2)-sp.cot(t)*sp.diff(y, t))/2)
    norm2 = 2*eigenvalue**2
    integral = sp.integrate(norm2*2*sp.pi*sp.sin(t), (t, 0, sp.pi))
    density = sp.trigsimp(norm2/integral)
    assert sp.simplify(density-15*sp.sin(t)**4/(32*sp.pi)) == 0
    assert density.subs(t, 0) == density.subs(t, sp.pi) == 0
    assert sp.integrate(density*2*sp.pi*sp.sin(t), (t, 0, sp.pi)) == 1
    left, right = [vertices(by_id(root, f"lobe-{k}")) for k in (1, 0)]
    (cx, cy) = right[0]
    assert left[0] == left[-1] == right[-1] == (cx, cy)
    radius = max(x-cx for x, _ in right)
    assert abs(cx-min(x for x, _ in left)-radius) < .001
    # Recover the polar angle from each drawn vertex, without the generator's angular grid.
    for x, y in left + right:
        r = hypot(x-cx, y-cy)
        if r > .05:
            assert abs(r/radius - ((x-cx)/r)**4) < .003
    assert (cx-radius, cy) in left and (cx+radius, cy) in right


def center(node):
    return float(node.get("cx")), float(node.get("cy"))


def endpoints(node):
    return [(float(node.get("x"+i)), float(node.get("y"+i))) for i in ("1", "2")]


def check_pericenters(root):
    check_planes(root)
    cx, cy = center(by_id(root, "plane-center"))
    inner, outer = (vertices(by_id(root, "well-"+edge)) for edge in ("inner", "outer"))
    crests = []
    for mode in ("lower", "upper"):
        curve, key = (by_id(root, prefix+mode) for prefix in ("carrier-", "key-"))
        dash = curve.get("stroke-dasharray")
        assert bool(dash) == (mode == "upper") and key.get("stroke-dasharray") == dash
        assert curve.get("stroke-width") == key.get("stroke-width")
        profile = []
        for (x, y), a, b in zip(vertices(curve), inner, outer, strict=True):
            dx, dy = b[0]-a[0], b[1]-a[1]
            length = hypot(dx, dy)
            assert abs((x-cx)*dy-(y-cy)*dx)/length < .002
            f = ((x-a[0])*dx+(y-a[1])*dy)/length**2
            assert 0 < f < 1
            profile.append(f)
        assert abs(sum(profile)/len(profile)-.5) < .001
        crests.append(sum(v > profile[i-1] and v > profile[(i+1) % len(profile)]
                          for i, v in enumerate(profile)))
    assert crests[1]-crests[0] == 2
    for name in ("exact", "near"):
        frame = by_id(root, name).find(SVG+"circle")
        cx, cy = center(frame)
        radius = float(frame.get("r"))
        points = [center(n) for n in by_id(root, name).findall(SVG+"circle")
                  if (n.get("id") or "").startswith(name+"-return-")]
        assert all(abs(dist(p, (cx, cy))-radius) < 1e-8 for p in points)
        phase = [atan2(cy-y, x-cx) for x, y in points]
        assert phase[0] == 0
        if name == "exact":
            assert len(phase) == 2 and abs(abs(phase[1])-pi) < 1e-8
            assert by_id(root, "label-exact-time-0").text == "0, 2"
        else:
            slip = remainder(phase[1]-5*pi, 2*pi)
            assert len(phase) == 3 and 0 < slip < pi/4
            assert abs(phase[2]-2*slip) < 1e-8
            arc = [atan2(cy-y, x-cx) for x, y in vertices(by_id(root, "advance-arc"))]
            assert min(b-a for a, b in pairwise(arc)) > 0 and abs(arc[-1]-phase[2]) < 1e-5
    fr, fp = sp.symbols("Omega_r Omega_phi", positive=True)
    period = 2*sp.pi/fr
    assert sp.simplify(2*(fp*period-5*sp.pi)-(2*fp-5*fr)*period) == 0


def check_observation(root):
    check_pattern(root)
    (left, t0), corner = endpoints(by_id(root, "initial-slice"))
    upper_edge, spatial = endpoints(by_id(root, "null-infinity"))
    assert t0 == spatial[1] and corner == spatial and upper_edge[0] < spatial[0]
    assert spatial[1]-upper_edge[1] == spatial[0]-upper_edge[0]
    well_top, well_base = endpoints(by_id(root, "well"))
    assert well_top[0] == well_base[0] and left < well_base[0] and well_base[1] == t0
    cuts = endpoints(by_id(root, "interval"))
    assert cuts[1][1] < cuts[0][1]  # Later retarded time is higher on future null infinity.
    for i, cut in enumerate(cuts, 1):
        source, arrival = endpoints(by_id(root, f"outgoing-{i}"))
        assert arrival == cut == center(by_id(root, f"cut-{i}"))
        assert upper_edge[0] < cut[0] < spatial[0] and cut[1]-upper_edge[1] == cut[0]-upper_edge[0]
        assert source[0] == well_base[0] and well_top[1] < source[1] < t0
        assert cut[0]-source[0] == source[1]-cut[1] > 0


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run without -O.")
    for stem, check in (("pericenter-returns", check_pericenters),
                        ("quadrupole-observation", check_observation)):
        check(ET.parse(HERE / (stem + ".svg")).getroot())
