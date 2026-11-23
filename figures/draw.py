"""Draw the two Letter figures as SVG and export them with Inkscape."""
import argparse
from math import cos, pi, radians, sin
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
HERE = Path(__file__).resolve().parent
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
INK, MID, GUIDE = "#222e34", "#54646a", "#bac6c9"


def element(parent, name, **attrs):
    return ET.SubElement(parent, f"{{{NS}}}{name}",
                         {k.replace("_", "-"): str(v) for k, v in attrs.items()})


def canvas(title, height):
    svg = ET.Element(f"{{{NS}}}svg", width="86mm", height=f"{86*height/336:.6f}mm",
                     viewBox=f"0 0 336 {height}")
    element(svg, "title").text = title
    return svg


def line(parent, x1, y1, x2, y2, color=MID, weight=1, **attrs):
    return element(parent, "line", x1=x1, y1=y1, x2=x2, y2=y2,
                   stroke=color, stroke_width=weight, **attrs)


def path(parent, points, color, weight=1, fill="none", close=False, **attrs):
    d = "M" + " L".join(f"{x:.4f},{y:.4f}" for x, y in points)
    return element(parent, "path", d=d + (" Z" if close else ""), fill=fill,
                   stroke=color, stroke_width=weight, stroke_linejoin="round", **attrs)


def label(parent, name, x, y, value, *, anchor="start", color=INK, size=11,
          math=False, bold=False, rotation=None, serif=False):
    attrs = {"transform": f"rotate({rotation} {x} {y})"} if rotation else {}
    element(parent, "text", id="label-" + name, x=x, y=y, fill=color,
            font_family="Times New Roman" if math or serif else "Arial",
            font_style="italic" if math else "normal", font_size=size,
            font_weight="bold" if bold else "normal", text_anchor=anchor, **attrs).text = value


def dot(parent, x, y, r, **attrs):
    return element(parent, "circle", cx=x, cy=y, r=r, **attrs)


def polar_outline(pattern, cx, cy, radius, color, weight, fill):
    for side in range(2):
        points = []
        for j in range(181):
            t = pi*(j/180+side)
            rho = radius*sin(t)**4
            points.append((cx+rho*sin(t), cy-rho*cos(t)))
        path(pattern, points, color, weight, fill, True, id=f"lobe-{side}")


def note(parent, name, x, y, value, **opts):
    label(parent, name, x, y, value, color="black", serif=True, **opts)


def plate(title):
    svg = canvas(title, 133)
    note(svg, "a", 4, 15, "(a)", size=12)
    note(svg, "b", 155, 15, "(b)", size=12)
    return svg


def print_axis(parent, x, ytop, ybottom):
    line(parent, x, ybottom, x, ytop, "black", .65)
    path(parent, [(x-2, ytop+4.2), (x, ytop), (x+2, ytop+4.2)], "black", .7)
    note(parent, "axis", x+6, ytop+4, "z", math=True, size=13)


def pericenters():
    svg = plate("Trapped modes and pericenter returns near the five-to-two resonance")
    g = element(svg, "g", id="representative-plane")
    cx, cy, radius, tilt, view = 52, 64, 43, pi/3, radians(44)
    angles = [2*pi*j/720 for j in range(720)]

    def orbit(t, scale):
        return (cx+radius*scale*(cos(tilt)*cos(t)*cos(view)-sin(t)*sin(view)),
                cy+radius*scale*sin(tilt)*cos(t))

    print_axis(g, cx, 13, 110)
    for name, scale in (("inner", .81), ("outer", 1.19)):
        path(g, [orbit(t, scale) for t in angles], "black", .45, close=True, id="well-"+name)
    # Both schematic wave traces oscillate about the same projected circle.
    for name, m, weight, style, x, legend in (
        ("lower", 9, 1.00, {}, 3, "(\u2113, n)"),
        ("upper", 11, 1.25, {"stroke_dasharray": "3.4 2.1"}, 62, "(\u2113 + 2, n \u2212 5)"),
    ):
        path(g, [orbit(t, 1+.12*cos(m*t)) for t in angles], "black", weight,
             close=True, id="carrier-"+name, **style)
        line(g, x, 121, x+16.5, 121, "black", weight, id="key-"+name, **style)
        note(g, "mode-"+name, x+21, 125, legend, math=True, size=12)
    dot(g, cx, cy, 1.4, fill="black", id="plane-center")
    normals = element(g, "g", id="normal-azimuths")
    nx, ny, nr = 117, 52, 12
    for j in range(5):
        phi = 2*pi*j/5
        x, y = nx+nr*sin(phi), ny-nr*cos(phi)
        line(normals, nx, ny, x, y, "black", .65)
        dot(normals, x, y, 1.8, fill="black")
    note(normals, "normals", nx, 83, "normals", anchor="middle")

    for cx, slip, name, caption in ((191, 0, "exact", "Resonance"),
                                    (286, radians(16), "near", "Nearby orbit")):
        cy, radius = 66, 30
        g = element(svg, "g", id=name)
        dot(g, cx, cy, radius, fill="none", stroke="black", stroke_width=.65)
        line(g, cx-radius, cy, cx+radius, cy, "black", .65, stroke_dasharray="3.2 2.8")

        def point(phi, r=radius):
            return cx+r*cos(phi), cy-r*sin(phi)

        for j in ((0, 1) if not slip else (0, 1, 2)):
            phi = j*(5*pi+slip)
            x, y = point(phi)
            if slip and j:
                line(g, cx, cy, x, y, "black", .9)
            dot(g, x, y, 2.25, fill="black", id=f"{name}-return-{j}")
        if slip:
            angle = 2*slip
            arc = [point(angle*j/24, 39) for j in range(25)]
            path(g, arc, "black", .8, id="advance-arc")
            tip = arc[-1]
            # Counterclockwise tangent in SVG's downward y coordinate.
            tx, ty = -sin(angle), -cos(angle)
            path(g, [(tip[0]-5*tx+2*ty, tip[1]-5*ty-2*tx), tip,
                     (tip[0]-5*tx-2*ty, tip[1]-5*ty+2*tx)], "black", .8)
            positions = [(cx+35, cy+16, "0", "middle"), (cx-35, cy+23, "1", "middle"),
                         (cx+22, cy-29, "2", "middle")]
        else:
            positions = [(cx+36, cy+3.5, "0, 2", "start"), (cx-36, cy+3.5, "1", "end")]
        for k, (x, y, txt, anchor) in enumerate(positions):
            note(g, f"{name}-time-{k}", x, y, txt, anchor=anchor, size=12)
        note(g, name+"-caption", cx, 125, caption, anchor="middle", size=12)
    return svg


def observation():
    svg = plate("Observation interval at null infinity and quadrupole energy on the sky")
    # Portion of a conformal diagram: future is up and outgoing rays rise rightward.
    g = element(svg, "g", id="spacetime")
    line(g, 34, 37, 34, 110, "black", .85, stroke_dasharray="4 2.4", id="well")
    note(g, "well", 34, 30, "Well", anchor="middle", size=12)
    line(g, 13, 110, 144, 110, "black", .85, id="initial-slice")
    note(g, "slice", 80, 125, "t = 0", anchor="middle", math=True, size=13)
    line(g, 53, 19, 144, 110, "black", .85, id="null-infinity")
    note(g, "infinity", 106, 66, "Null infinity", size=12)
    cuts = [(86, 52), (69, 35)]
    line(g, *cuts[0], *cuts[1], "black", 1.9, id="interval")
    note(g, "window", 66, 58, "I", math=True, size=13)
    for i, ((x, y), txt) in enumerate(zip(cuts, ("u\u2081", "u\u2082")), 1):
        emit = y+x-34
        line(g, 34, emit, x, y, "black", .8, id=f"outgoing-{i}")
        dot(g, x, y, 1.9, fill="black", id=f"cut-{i}")
        note(g, f"time-{i}", x+8, y-2, txt, math=True, size=13)
    p = element(svg, "g", id="polar-section")
    cx, cy, radius = 251, 66, 71
    print_axis(p, cx, 14, 106)
    line(p, cx-radius-5, cy, cx+radius+5, cy, "black", .6, stroke_dasharray="3.2 2.8")
    polar_outline(p, cx, cy, radius, "black", 1.15, "none")
    angle = radians(63)
    rho = radius*sin(angle)**4
    line(p, cx, cy, cx+rho*sin(angle), cy-rho*cos(angle), "black", .65)
    path(p, [(cx+14*sin(angle*j/24), cy-14*cos(angle*j/24)) for j in range(25)], "black", .65)
    note(p, "theta", cx+10, cy-21, "\u03d1", math=True, size=13)
    note(p, "pattern", cx, 125, "sin\u2074 \u03d1", anchor="middle", size=13)
    return svg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inkscape", default=shutil.which("inkscape"))
    parser.add_argument("--figure", help="export only the named figure (file stem)")
    opts = parser.parse_args()
    if not opts.inkscape:
        parser.error("supply --inkscape PATH or add Inkscape to PATH")
    figures = (
        ("pericenter-returns", pericenters), ("quadrupole-observation", observation))
    if opts.figure:
        if opts.figure not in dict(figures):
            parser.error("choose a figure from: "+", ".join(dict(figures)))
        figures = [(opts.figure, dict(figures)[opts.figure])]
    for stem, make in figures:
        svg = make()
        ET.indent(svg)
        src = HERE / (stem+".svg")
        ET.ElementTree(svg).write(src, encoding="utf-8", xml_declaration=True)
        subprocess.run([opts.inkscape, str(src), "--export-area-page",
                        f"--export-filename={src.with_suffix('.pdf')}"], check=True)


if __name__ == "__main__":
    main()
