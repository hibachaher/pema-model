import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import SA_NAMES

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})
ACC = "#1d5d72"; RED = "#a63a2e"; AMB = "#b07a1c"; GRN = "#2f7d4f"; INK = "#1b2a33"; GREY = "#8a99a3"
R = {}

# ---------- Sobol indices (Saltelli 2010 first-order, Jansen total) with bootstrap CIs
z = np.load("out/sa.npz"); X, Y, n, d = z["X"], z["Y"], int(z["n"]), int(z["d"])
labels = {"kappa": "load conversion κ", "p_neg": "stressor rate", "alpha": "dissipation α",
          "rho0": "recovery ρ₀", "omega_scale": "Ω magnitude", "theta_mu": "tolerance θ",
          "eta": "appraisal weight η"}
def sobol(yA, yB, yAB):
    V = np.var(np.concatenate([yA, yB]), ddof=1)
    S1 = np.mean(yB[:, None] * (yAB - yA[:, None]), 0) / V
    ST = 0.5 * np.mean((yA[:, None] - yAB) ** 2, 0) / V
    return S1, ST
rng = np.random.default_rng(3)
out_names = ["turnover", "ocb"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
for k, oname in enumerate(out_names):
    y = Y[:, k]
    yA, yB = y[:n], y[n:2 * n]
    yAB = np.column_stack([y[(2 + j) * n:(3 + j) * n] for j in range(d)])
    S1, ST = sobol(yA, yB, yAB)
    bs1, bst = [], []
    for _ in range(500):
        i = rng.integers(0, n, n)
        a, b = sobol(yA[i], yB[i], yAB[i]); bs1.append(a); bst.append(b)
    ci1 = np.percentile(bs1, [2.5, 97.5], 0); cit = np.percentile(bst, [2.5, 97.5], 0)
    R[oname] = {SA_NAMES[j]: dict(S1=float(S1[j]), ST=float(ST[j]),
                                  S1_ci=list(map(float, ci1[:, j])), ST_ci=list(map(float, cit[:, j])))
                for j in range(d)}
    order = np.argsort(ST)
    yy = np.arange(d)
    ax = axs[k]
    ax.barh(yy + 0.18, ST[order], height=0.34, color=ACC, label="total effect $S_T$",
            xerr=np.abs(cit[:, order] - ST[order]), error_kw=dict(lw=0.6, capsize=1.5))
    if k == 0:
        ax.barh(yy - 0.18, np.clip(S1[order], 0, None), height=0.34, color="#9cc3cf", label="first order $S_1$",
                xerr=np.abs(np.clip(ci1[:, order], 0, None) - np.clip(S1[order], 0, None)), error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_yticks(yy); ax.set_yticklabels([labels[SA_NAMES[j]] for j in order])
    ax.set_xlim(0, 1)
    ax.set_title(("a  Annual turnover rate" if k == 0 else "b  Mean OCB propensity (S$_T$)"), loc="left",
                 fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Sobol index")
axs[0].legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig("fig/fig4_sobol.png", dpi=600, facecolor="white")
R["sa_runs"] = int(len(Y)); R["sa_turnover_range"] = [float(Y[:, 0].min()), float(Y[:, 0].max())]

# ---------- Interventions
r = json.load(open("out/scen.json"))
names = []
for x in r:
    if x[0] not in names: names.append(x[0])
S = {}
for nm in names:
    a = np.array([x[1:] for x in r if x[0] == nm], float)
    S[nm] = dict(turn_ci95_mean=[a[:,0].mean()-1.96*a[:,0].std(ddof=1)/np.sqrt(len(a)), a[:,0].mean()+1.96*a[:,0].std(ddof=1)/np.sqrt(len(a))], turn=a[:, 0].mean(), turn_sd=a[:, 0].std(ddof=1), turn_ci=list(np.percentile(a[:, 0], [2.5, 97.5])),
                 ocb=a[:, 1].mean(), cwb=a[:, 2].mean(),
                 tdays=a[:, 3].mean() / 300 / 1000 * 250 * 100)   # treated person-days / 100 employees / year
base = S["Baseline"]["turn"]
for nm in names:
    S[nm]["exits_avoided_per100_yr"] = (base - S[nm]["turn"]) * 100
    S[nm]["avoided_per_1000_tdays"] = (S[nm]["exits_avoided_per100_yr"] / S[nm]["tdays"] * 1000) if S[nm]["tdays"] > 0 else None
R["scenarios"] = S
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.25, 1]})
cols = [GREY, GRN, AMB, RED, ACC]
short = ["Baseline", "Stressor\nreduction −20%", "Universal\nrecovery prog.", "Late support\n(CWB onset)", "Early support\n(OCB decline)"]
y = np.arange(len(names))[::-1]
axs[0].barh(y, [S[n]["turn"] * 100 for n in names], color=cols,
            xerr=[1.96 * S[n]["turn_sd"] / np.sqrt(30) * 100 for n in names],
            error_kw=dict(lw=0.7, capsize=2))
axs[0].set_yticks(y); axs[0].set_yticklabels(short, fontsize=7)
axs[0].set_xlabel("Annual turnover (%), mean ± 95% CI")
axs[0].set_title("a  Turnover under each policy", loc="left", fontsize=8.5, fontweight="bold")
for yi, n in zip(y, names):
    axs[0].text(S[n]["turn"] * 100 + 1.96 * S[n]["turn_sd"] / np.sqrt(30) * 100 + 0.4, yi, f"{S[n]['turn']*100:.1f}", va="center", fontsize=7)
sel = ["Universal recovery programme", "Late support (CWB onset)", "Early support (OCB drop)"]
td = [S[n]["tdays"] for n in sel]; av = [S[n]["exits_avoided_per100_yr"] for n in sel]
axs[1].scatter(td, av, s=40, color=[AMB, RED, ACC], zorder=3)
for t_, a_, lab, off in zip(td, av, ["Universal programme", "Late support", "Early support"], [(-30, 8), (8, -12), (-10, 8)]):
    axs[1].annotate(lab, (t_, a_), textcoords="offset points", xytext=off, fontsize=7)
axs[1].set_xscale("log"); axs[1].set_xlim(150, 80000); axs[1].set_ylim(0, max(av) * 1.35)
axs[1].set_xlabel("Support effort (person-days per 100 staff per year)")
axs[1].set_ylabel("Exits avoided per 100 staff per year")
axs[1].set_title("b  Effort vs. benefit", loc="left", fontsize=8.5, fontweight="bold")
fig.tight_layout(); fig.savefig("fig/fig5_interventions.png", dpi=600, facecolor="white")
json.dump(R, open("out/sa_scen_summary.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
