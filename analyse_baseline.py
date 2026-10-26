import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})
ACC = "#1d5d72"; RED = "#a63a2e"; AMB = "#b07a1c"; GRN = "#2f7d4f"; INK = "#1b2a33"; GREY = "#8a99a3"

res = list(np.load("out/baseline.npy", allow_pickle=True))
R = {}
tr = np.array([r["turnover"] for r in res])
R["turnover_mean"] = tr.mean(); R["turnover_ci"] = list(np.percentile(tr, [2.5, 97.5]))
R["ocb_mean"] = float(np.mean([r["mean_ocb"] for r in res]))
R["cwb_mean"] = float(np.mean([r["mean_cwb"] for r in res]))
auc = np.array([r["auc"] for r in res])
R["auc_AL"] = [float(np.nanmean(auc[:, 0])), float(np.nanstd(auc[:, 0]))]
R["auc_S"] = [float(np.nanmean(auc[:, 1])), float(np.nanstd(auc[:, 1]))]
R["auc_OCB"] = [float(np.nanmean(auc[:, 2])), float(np.nanstd(auc[:, 2]))]
d = auc[:, 0] - auc[:, 1]; d = d[~np.isnan(d)]
R["auc_diff_wilcoxon_p"] = float(stats.wilcoxon(d).pvalue); R["auc_diff_mean"] = float(d.mean())

# H4 ordering
order = np.vstack([r["order"] for r in res if len(r["order"])])
both = order[(order[:, 0] >= 0) & (order[:, 1] >= 0)]
R["n_overload_exits"] = int(len(order))
R["frac_both_signals"] = float(len(both) / len(order))
R["frac_ocb_before_cwb"] = float(np.mean(both[:, 0] < both[:, 1]))
R["lead_ocb_to_cwb_median"] = float(np.median(both[:, 1] - both[:, 0]))
R["lead_ocb_to_cwb_iqr"] = list(map(float, np.percentile(both[:, 1] - both[:, 0], [25, 75])))
R["lead_cwb_to_exit_median"] = float(np.median(both[:, 2] - both[:, 1]))
R["lead_cwb_to_exit_iqr"] = list(map(float, np.percentile(both[:, 2] - both[:, 1], [25, 75])))
R["lead_ocb_to_exit_median"] = float(np.median(both[:, 2] - both[:, 0]))

# survival / tenure by traits (pooled spells): cols O C E A N tenure censored cause
sp = np.vstack([r["spells"] for r in res])
O, C, E, A, N, ten, cen, cause = sp.T
def km(t, c):
    order_ = np.argsort(t); t = t[order_]; ev = 1 - c[order_]
    uniq = np.unique(t[ev == 1]); S = 1.0; xs = [0]; ys = [1.0]
    n_at = len(t)
    for u in uniq:
        n_risk = (t >= u).sum(); d_ = ((t == u) & (ev == 1)).sum()
        S *= 1 - d_ / n_risk; xs.append(u); ys.append(S)
    return np.array(xs), np.array(ys)
def median_surv(t, c):
    x, y = km(t, c)
    return float(x[np.argmax(y <= 0.5)]) if (y <= 0.5).any() else np.nan

groups = {}
lowC, highC = C < np.quantile(C, 1/3), C > np.quantile(C, 2/3)
lowN, highN = N < np.quantile(N, 1/3), N > np.quantile(N, 2/3)
for name, m in {"Resilient (high C, low N)": highC & lowN, "High C, high N": highC & highN,
                "Low C, low N": lowC & lowN, "Vulnerable (low C, high N)": lowC & highN}.items():
    x_, y_ = km(ten[m], cen[m]); groups[name] = (float(1 - y_[x_ <= 999][-1]), median_surv(ten[m], cen[m]), int(m.sum()))
R["median_tenure_groups"] = groups

# Spearman of tenure among completed stress exits with traits
ex = (cen == 0) & (cause == 1)
R["spearman_tenure"] = {k: float(stats.spearmanr(v[ex], ten[ex])[0]) for k, v in
                        zip(["O", "C", "E", "A", "N"], [O, C, E, A, N])}
# 1-year exit probability by C decile ratio
def p_exit_1y(m):
    t, c = ten[m], cen[m]
    x, y = km(t, c); return 1 - y[x <= 250][-1]
R["p_exit_250d_lowC_highN"] = float(p_exit_1y(lowC & highN))
R["p_exit_250d_highC_lowN"] = float(p_exit_1y(highC & lowN))

# ---------------- Figure 2: KM survival by profile + tenure heat map
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
cols = [GRN, ACC, AMB, RED]
for (name, m), col in zip({"Resilient (high C, low N)": highC & lowN, "High C, high N": highC & highN,
                           "Low C, low N": lowC & lowN, "Vulnerable (low C, high N)": lowC & highN}.items(), cols):
    x, y = km(ten[m], cen[m]); axs[0].step(x, y, where="post", color=col, lw=1.4, label=name)
axs[0].axhline(0.5, color=GREY, lw=0.6, ls=":")
axs[0].set_xlabel("Tenure (working days)"); axs[0].set_ylabel("Probability still employed")
axs[0].set_xlim(0, 1000); axs[0].set_ylim(0, 1.02); axs[0].legend(frameon=False, fontsize=6.8, loc="lower left")
axs[0].set_title("a  Kaplan–Meier retention by profile", loc="left", fontsize=8.5, fontweight="bold")
# heat map of median tenure on C x N quintiles
qs = np.linspace(0, 1, 6)
cb = np.quantile(C, qs); nb = np.quantile(N, qs)
H = np.full((5, 5), np.nan)
for i in range(5):
    for j in range(5):
        m = (C >= cb[i]) & (C <= cb[i + 1]) & (N >= nb[j]) & (N <= nb[j + 1])
        x_, y_ = km(ten[m], cen[m]); H[j, i] = 100 * (1 - y_[x_ <= 999][-1])
im = axs[1].imshow(H, origin="lower", cmap="OrRd", vmin=0, vmax=max(60, np.nanmax(H)), aspect="auto")
for i in range(5):
    for j in range(5):
        v = H[j, i]; txt = f"{v:.0f}%"
        axs[1].text(i, j, txt, ha="center", va="center", fontsize=6.3, color="white" if v > 35 else INK)
axs[1].set_xticks(range(5)); axs[1].set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
axs[1].set_yticks(range(5)); axs[1].set_yticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
axs[1].set_xlabel("Conscientiousness quintile"); axs[1].set_ylabel("Neuroticism quintile")
axs[1].set_title("b  Exit probability within 1,000 days", loc="left", fontsize=8.5, fontweight="bold")
cbar = fig.colorbar(im, ax=axs[1], fraction=0.05, pad=0.03); cbar.ax.tick_params(labelsize=6.5)
fig.tight_layout(); fig.savefig("fig/fig2_retention.png", dpi=600, facecolor="white")
R["heat_exit1000_pct"] = H.tolist()

# ---------------- Figure 3: event-aligned erosion sequence
al = [r["aligned_mean"] * r["n_aligned"] for r in res if r["aligned_mean"] is not None]
n_al = sum(r["n_aligned"] for r in res if r["aligned_mean"] is not None)
M = np.sum(al, axis=0) / n_al          # (4, W+1): L, S, OCB, CWB
R["n_aligned_total"] = int(n_al)
W = M.shape[1] - 1; x = np.arange(-W, 1)
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.axvspan(x[np.argmax(M[0] >= 0.45)], x[np.argmax(M[0] >= 0.75)], color="#f7ecd9", lw=0)
ax.axvspan(x[np.argmax(M[0] >= 0.75)], 0, color="#f6e3e0", lw=0)
ax.plot(x, M[0], color=INK, lw=1.6, label="Allostatic load  $A_L/\\theta$")
ax.plot(x, M[1], color=ACC, lw=1.2, ls="--", label="Job satisfaction  S(t)")
ax.plot(x, M[2], color=GRN, lw=1.4, label="OCB propensity")
ax.plot(x, M[3], color=RED, lw=1.4, label="CWB propensity")
ax.axhline(0.45, color=GREY, lw=0.5, ls=":"); ax.axhline(0.75, color=GREY, lw=0.5, ls=":")
ax.text(x[np.argmax(M[0] >= 0.45)] + 4, 1.04, "erosion phase", fontsize=7, color=AMB)
ax.text(x[np.argmax(M[0] >= 0.75)] + 4, 1.04, "deviance phase", fontsize=7, color=RED)
ax.set_xlim(-W, 0); ax.set_ylim(0, 1.1)
ax.set_xlabel("Days before resignation (event time, overload exits only)"); ax.set_ylabel("Mean value")
ax.legend(frameon=False, fontsize=7, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); fig.savefig("fig/fig3_erosion.png", dpi=600, facecolor="white")
R["erosion_enter_day"] = int(x[np.argmax(M[0] >= 0.45)]); R["deviance_enter_day"] = int(x[np.argmax(M[0] >= 0.75)])
R["ocb_at_minus400"] = float(M[2, 0]); R["ocb_at_exit"] = float(M[2, -1])
R["cwb_at_minus400"] = float(M[3, 0]); R["cwb_at_exit"] = float(M[3, -1])

# ---------------- H1 shock
sh = np.load("out/shock.npz")
Nn = sh["psi"][:, 4]; rec = sh["rec"]; drop = sh["drop"]
R["h1_rec_spearman"] = float(stats.spearmanr(Nn, rec)[0])
R["h1_drop_spearman"] = float(stats.spearmanr(Nn, drop)[0])
lo, hi = Nn < np.quantile(Nn, 0.2), Nn > np.quantile(Nn, 0.8)
R["h1_rec_lowN"] = float(np.median(rec[lo])); R["h1_rec_highN"] = float(np.median(rec[hi]))
R["h1_drop_lowN"] = float(np.median(drop[lo])); R["h1_drop_highN"] = float(np.median(drop[hi]))
json.dump(R, open("out/baseline_summary.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
