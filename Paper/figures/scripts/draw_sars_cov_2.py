"""Rebuild the paper's original, editable vector schematics.

Run from any directory: python3 figures/scripts/draw_sars_cov_2.py
Outputs PDF and SVG figures, plus PNG previews, into figures/.
The virion is a conceptual illustration, not a molecular or scale model.

Biological references:
https://www.ncbi.nlm.nih.gov/Structure/SARS-CoV-2.html
https://arxiv.org/html/2501.14746v1 (genome overview, Fig. 1)
Layout follows the reference illustration supplied by the author.
"""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "quspike-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})

INK = "#282C32"
RED, RED_EDGE = "#C34E53", "#793036"
PINK, PINK_EDGE = "#F1DCDD", "#B48C92"
TEAL, TEAL_EDGE = "#62BBC5", "#2C7582"
PURPLE, PURPLE_EDGE = "#9674B9", "#62487E"
BLUE = "#376A9C"


def canvas(width, height):
    fig = plt.figure(figsize=(width / 72, height / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height), aspect="equal")
    ax.axis("off")
    return fig, ax


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(ROOT / f"{name}.{ext}", dpi=150, facecolor="white")
    plt.close(fig)


def spike(ax, x, y, angle, scale=1, z=4):
    """A stalk with three rounded lobes, consistently used in both views."""
    transform = Affine2D().scale(scale).rotate_deg(angle - 90).translate(x, y) + ax.transData
    points = [(-2.4, 0), (-3, 10), (-7, 15), (-13, 17), (-15, 23),
              (-13, 30), (-7, 31), (-4, 29), (-3, 36), (3, 36),
              (6, 31), (10, 33), (16, 29), (15, 22), (9, 17),
              (3, 10), (2.4, 0), (-2.4, 0)]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * 16 + [MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(points, codes), facecolor=RED, edgecolor=RED_EDGE,
                           lw=1.35, transform=transform, zorder=z))
    for px, py, r in [(-7, 24, 4), (1, 29, 4), (9, 25, 3.5)]:
        ax.add_patch(Circle((px, py), r, facecolor="#DB7578", edgecolor="none",
                            transform=transform, zorder=z + .1))


def membrane(ax, x, y, angle, scale=1, z=5):
    transform = Affine2D().scale(scale).rotate_deg(angle - 90).translate(x, y) + ax.transData
    for dx in (-3.2, 3.2):
        points = [(dx, -8), (dx + 3, -2), (dx, 2), (dx - 2, 6), (dx, 11)]
        patch = PathPatch(MplPath(points, [MplPath.MOVETO] + [MplPath.CURVE3] * 4),
                          facecolor="none", edgecolor=TEAL_EDGE, lw=4.8,
                          transform=transform, zorder=z)
        ax.add_patch(patch)
        ax.add_patch(PathPatch(patch.get_path(), facecolor="none", edgecolor=TEAL,
                               lw=2.7, transform=transform, zorder=z + .1))


def envelope(ax, x, y, angle, scale=1, z=5):
    transform = Affine2D().scale(scale).rotate_deg(angle - 90).translate(x, y) + ax.transData
    ax.add_patch(Rectangle((-2.6, -7), 5.2, 15, facecolor=PURPLE,
                           edgecolor=PURPLE_EDGE, lw=1.1, transform=transform, zorder=z))


def polar(cx, cy, r, degrees):
    theta = np.deg2rad(degrees)
    return cx + r * np.cos(theta), cy + r * np.sin(theta)


def leader(ax, points, label, text_x, text_y):
    ax.plot(*np.array(points).T, color=INK, lw=1.05, zorder=12)
    ax.add_patch(Circle(points[0], 2.3, facecolor=INK, edgecolor="none", zorder=13))
    ax.text(text_x, text_y, label, fontsize=13.5, color=INK, va="center", zorder=13)


def virion():
    fig, ax = canvas(930, 350)
    cx, cy, r = 177, 180, 112
    # Front exterior view: a continuous closed envelope with surface proteins.
    ax.add_patch(Circle((cx, cy), r, facecolor=PINK, edgecolor=PINK_EDGE, lw=1.4))
    ax.add_patch(Ellipse((cx - 11, cy + 15), 192, 180, facecolor="#F9EDEE",
                         edgecolor="none", zorder=1.1))
    for a in np.arange(4, 360, 25):
        x, y = polar(cx, cy, r - 1, a)
        spike(ax, x, y, a, .86)
    # Trimer heads seen face-on on the visible surface.
    for x, y, turn, scale in [(125, 240, 15, 1), (212, 249, 50, .91),
                              (91, 186, -20, .84), (171, 186, 40, 1.05),
                              (244, 193, -35, .9), (118, 125, 50, .89),
                              (216, 115, 0, .92)]:
        t = Affine2D().scale(scale).rotate_deg(turn).translate(x, y) + ax.transData
        for dx, dy in [(-7, 4), (6, 6), (0, -7)]:
            ax.add_patch(Circle((dx, dy), 9, facecolor=RED, edgecolor=RED_EDGE,
                                lw=1.2, transform=t, zorder=5))
            ax.add_patch(Circle((dx - 2, dy + 2), 4, facecolor="#DB7578",
                                edgecolor="none", transform=t, zorder=5.1))
    for x, y in [(139, 272), (183, 272), (106, 220), (152, 219), (244, 231),
                  (76, 163), (125, 166), (208, 210), (207, 164), (264, 158),
                  (155, 131), (180, 97), (239, 139), (148, 95)]:
        ax.add_patch(Circle((x, y), 3.3, facecolor=TEAL, edgecolor=TEAL_EDGE, lw=1, zorder=3))
        ax.add_patch(Circle((x + 7, y + 4), 2.6, facecolor=TEAL, edgecolor=TEAL_EDGE, lw=.9, zorder=3))
    for x, y in [(184, 239), (100, 148), (253, 212), (158, 150)]:
        ax.add_patch(Ellipse((x, y), 7, 4.5, angle=35, facecolor=PURPLE,
                             edgecolor=PURPLE_EDGE, lw=1, zorder=3))

    # Oblique cut: the remaining spherical shell is visible to the left of
    # a tilted elliptical section plane, with the double membrane at its rim.
    sx, sy, sr = 517, 180, 112
    ax.add_patch(Circle((sx, sy), sr, facecolor="#DDC0C4", edgecolor=PINK_EDGE, lw=1.4))
    ax.add_patch(Ellipse((sx - 18, sy + 14), 180, 187, facecolor=PINK, edgecolor="none", zorder=1.1))
    for a in (66, 93, 120, 147, 174, 201, 228, 255, 282):
        x, y = polar(sx, sy, sr - 1, a)
        spike(ax, x, y, a, .86, z=2)
    for a in (80, 132, 185, 237, 270):
        x, y = polar(sx, sy, sr - 1, a)
        membrane(ax, x, y, a, .65, z=3)
    cut = Affine2D().rotate_deg(-23).translate(548, 179)
    outer = Ellipse((0, 0), 151, 226, facecolor="#E6D4D5", edgecolor="#9D8589",
                    lw=1.4, transform=cut + ax.transData, zorder=5)
    ax.add_patch(outer)
    ax.add_patch(Ellipse((0, 0), 137, 211, facecolor="#FCFDFE", edgecolor="#AA9598",
                         lw=1.15, transform=cut + ax.transData, zorder=5.1))

    # Stylized continuous RNA, using smooth Bezier segments in the cut plane.
    controls = [(-19, 77), (-68, 67), (-53, 16), (-17, 24),
                (32, 34), (42, 88), (5, 73),
                (-22, 61), (19, 18), (41, 21),
                (71, 25), (57, -29), (26, -17),
                (-5, -5), (-48, 2), (-43, -29),
                (-39, -61), (8, -49), (2, -25),
                (-7, 6), (-66, -43), (-33, -69),
                (-5, -96), (46, -58), (20, -48)]
    rna = MplPath(controls, [MplPath.MOVETO] + [MplPath.CURVE4] * (len(controls) - 1))
    ax.add_patch(PathPatch(rna, facecolor="none", edgecolor=BLUE, lw=2.6,
                           transform=cut + ax.transData, zorder=6))
    # N-protein beads lie on the same RNA path.
    bead_points = []
    for i in range(0, len(controls) - 1, 3):
        p0, p1, p2, p3 = np.array(controls[i:i + 4])
        for t in (.23, .72):
            p = (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t*t*p2 + t**3*p3
            bead_points.append(p)
            x, y = cut.transform(p)
            ax.add_patch(Circle((x, y), 5.2, facecolor="#DAEFF0", edgecolor=TEAL_EDGE, lw=1.2, zorder=7))

    # Symbols are embedded in the cut rim rather than floating inside the virion.
    def rim(a):
        t = np.deg2rad(a)
        return cut.transform((72 * np.cos(t), 109 * np.sin(t)))

    mx, my = rim(26)
    membrane(ax, mx, my, 27, .78, z=8)
    ex, ey = rim(6)
    envelope(ax, ex, ey, 0, .86, z=8)
    for a in (70, 114, 158, 201, 244, 290):
        px, py = rim(a)
        membrane(ax, px, py, a - 23, .64, z=8)
    for a in (48, 223):
        px, py = rim(a)
        envelope(ax, px, py, a - 23, .75, z=8)

    # Aligned labels and fine elbow leaders match a textbook figure layout.
    tip = polar(sx, sy, sr + 25, 66)
    leader(ax, [tip, (679, 309), (710, 309)], "Spike protein (S)", 722, 309)
    leader(ax, [(mx + 3, my + 2), (677, 258), (710, 258)], "Membrane protein (M)", 722, 258)
    leader(ax, [(ex + 2, ey), (681, 209), (710, 209)], "Envelope protein (E)", 722, 209)
    nx, ny = cut.transform(bead_points[7])
    leader(ax, [(nx, ny), (668, 159), (710, 159)], "Nucleocapsid protein (N)", 722, 159)
    # A bare segment, not an N bead, is selected for the RNA leader.
    rx, ry = cut.transform(controls[-1])
    leader(ax, [(rx, ry), (675, 109), (710, 109)], "Single-stranded RNA", 722, 109)
    lx, ly = rim(-58)
    leader(ax, [(lx, ly), (670, 60), (710, 60)], "Lipid bilayer", 722, 60)

    ax.text(177, 12, "(a) Front exterior view", fontsize=14, ha="center", color=INK)
    ax.text(526, 12, "(b) Oblique cutaway view", fontsize=14, ha="center", color=INK)
    save(fig, "sars-cov-2-structure-2d")


def genome():
    fig, ax = canvas(520, 174)
    left, right = 32, 500
    def x(kb):
        return left + (right - left) * kb / 30

    ax.text(left, 157, "SARS-CoV-2 genome", fontsize=12, weight="bold", color=INK)
    ax.text(right, 157, "Approx. 30 kb", fontsize=10, ha="right", color=INK)
    ax.annotate("", xy=(right + 3, 129), xytext=(left, 129),
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 1})
    for k in range(0, 31, 5):
        ax.plot([x(k), x(k)], [126, 132], color=INK, lw=.8)
        ax.text(x(k), 139, str(k), fontsize=8.7, ha="center", color=INK)
    ax.text(12, 129, "5'", fontsize=10, va="center", color=INK)
    ax.text(508, 129, "3'", fontsize=10, va="center", color=INK)
    ax.plot([left, right], [105, 105], color="#D6D9DC", lw=1.2, zorder=0)

    for a, b, name in [(.27, 13.48, "ORF1a"), (13.44, 21.55, "ORF1b")]:
        ax.add_patch(Rectangle((x(a), 94), x(b)-x(a), 22,
                               facecolor="#EFC6A7", edgecolor="white", lw=1))
        ax.text((x(a)+x(b))/2, 105, name, fontsize=10, ha="center", va="center", color=INK)
    for a, b, name, fill in [(21.56, 25.38, "S", "#8EB779"),
                              (26.25, 26.47, "E", TEAL),
                              (26.52, 27.19, "M", TEAL),
                              (28.27, 29.53, "N", TEAL)]:
        ax.add_patch(Rectangle((x(a), 94), x(b)-x(a), 22,
                               facecolor=fill, edgecolor=INK, lw=.6))
        center = (x(a)+x(b))/2
        if name == "E":
            ax.plot([center, center-6], [94, 84], lw=.7, color=INK)
            ax.text(center-7, 79, name, fontsize=8.3, ha="center", va="center", color=INK)
        elif name == "M":
            ax.plot([center, center+5], [94, 84], lw=.7, color=INK)
            ax.text(center+6, 79, name, fontsize=8.3, ha="center", va="center", color=INK)
        else:
            ax.text(center, 105, name, fontsize=10, ha="center", va="center", color=INK)

    ax.plot([x(.27), x(.27), x(21.55), x(21.55)], [87, 82, 82, 87], color="#B3825D", lw=.8)
    ax.text((x(.27)+x(21.55))/2, 70, "Nonstructural protein-coding region", fontsize=8.7,
            color="#76553F", ha="center")
    ax.plot([x(21.56), x(21.56), x(25.38), x(25.38)], [87, 79, 79, 87], color="#557844", lw=.9)
    ax.text((x(21.56)+x(25.38))/2, 66, "Spike (S)", fontsize=9.5, color="#456938", ha="center")
    ax.text((x(21.56)+x(25.38))/2, 53, "~21-25 kb", fontsize=8.5, color="#456938", ha="center")
    # Accessory ORFs are a distinct track, preserving the legible main map.
    ax.text(left, 29, "Accessory ORFs", fontsize=8.2, color="#666C73", va="center")
    for a, b in [(25.39, 26.22), (27.20, 27.39), (27.39, 27.76),
                  (27.76, 27.89), (27.89, 28.26)]:
        ax.add_patch(Rectangle((x(a), 24), x(b)-x(a), 10,
                               facecolor="#BCC1C7", edgecolor="white", lw=.5))
    ax.text(right, 8, "Selected regions; boundaries are approximate.", fontsize=7.5,
            color="#666C73", ha="right")
    save(fig, "sars-cov-2-genome")


if __name__ == "__main__":
    virion()
    genome()
