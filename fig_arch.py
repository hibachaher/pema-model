import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")

INK = "#1b2a33"; ACC = "#1d5d72"; SOFT = "#e2eef2"; WARM = "#f6e7d8"; GREY = "#eef1f3"; RED = "#a63a2e"

def box(x, y, w, h, title, body, fc, ec=INK, bold=True):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                       fc=fc, ec=ec, lw=0.9)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.6,
            fontweight="bold" if bold else "normal", color=INK)
    ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=6.5, color=INK,
            linespacing=1.35)

def arrow(x1, y1, x2, y2, color=INK, ls="-", rad=0.0, lw=1.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                        color=color, lw=lw, linestyle=ls, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)

# layer labels
for x, lab in [(2, "PERCEPTION"), (27, "APPRAISAL"), (52, "INTERNAL STATE"), (77, "BEHAVIOUR")]:
    ax.text(x, 62.5, lab, fontsize=7, color=ACC, fontweight="bold")

box(1, 36, 21, 23, "Workplace stimuli",
    "Events (self/others):\nworkload, pay,\npromotion\n\nAgent actions\npraise, blame, help", GREY)
box(26, 36, 22, 23, "OCC appraisal",
    "desirability,\npraise/blameworthiness\n\n20 emotion intensities\n$I_e = u_s \\cdot g_e(\\Psi)$", SOFT)
box(26, 8, 22, 21, "Personality $\\Psi$",
    "Big Five O, C, E, A, N\n\nsensitivity matrix $\\Omega_\\Psi$\n(amplify / inhibit)\n$\\delta = \\alpha/(1+N)$", WARM)
box(51, 36, 23, 23, "Affective dynamics",
    "emotional state IES(t)\nlingering affect NA, PA\n\njob satisfaction S(t)\nset-point + inertia $\\delta$", SOFT)
box(51, 8, 23, 21, "Allostatic load",
    "$A_L(t)$: cumulative wear\nbuffered by (1 − C)\nrecovery from PA\n\nphase  $A_L/\\theta$", SOFT)
box(77, 36, 22, 23, "Fuzzy behaviour",
    "Mamdani FIS\ninputs S, $A_L/\\theta$, A\n\nOCB  ↓   CWB  ↑", SOFT)
box(77, 8, 22, 21, "Turnover",
    "fuzzy overload hazard\n$A_L \\geq \\theta_{turnover}$\n\nleaver replaced\nby new hire", "#f6e3e0", ec=RED)

arrow(22.6, 47.5, 25.4, 47.5)
arrow(48.6, 47.5, 50.4, 47.5)
arrow(74.6, 47.5, 76.4, 47.5)
arrow(37, 29.6, 37, 35.4, color="#9a6a2a")            # personality -> appraisal
arrow(48.6, 18, 50.4, 18, color="#9a6a2a")            # personality -> AL (1-C)
arrow(48.6, 24, 52, 35.4, color="#9a6a2a", rad=-0.1)  # personality -> delta
arrow(62, 35.4, 62, 29.6)                              # NA -> AL
arrow(74.6, 22, 76.4, 40, rad=0.15)                    # AL -> behaviour
arrow(74.6, 14, 76.4, 14, color=RED)                   # AL -> turnover
arrow(90, 35.4, 90, 29.6, color=RED, ls="--")          # behaviour precedes exit

# cybernetic regulator loop
ax.add_patch(FancyBboxPatch((23.5, 0.6), 51, 4.6, boxstyle="round,pad=0.3,rounding_size=1",
                            fc="white", ec=ACC, lw=0.9, ls="--"))
ax.text(49, 2.9, "HR regulator: observe OCB decline → targeted recovery support",
        ha="center", va="center", fontsize=6.3, color=ACC)
arrow(75.7, 35.8, 75.2, 4.4, color=ACC, ls="--")
arrow(66, 5.4, 66, 7.6, color=ACC, ls="--")

fig.savefig("fig/fig1_architecture.png", dpi=600, bbox_inches="tight", facecolor="white")
print("ok")
