"""Hazard-specification ablation (robustness check for Section 6.3 / 7.3).
Exit hazard h = h0 + h_max * [(1-w) * mu_load + w * mu_sat], with mu_sat = clip((s0 - S)/0.15, 0, 1).
w = 0   : baseline (exit driven by allostatic load only)
w = 0.5 : half load-driven, half satisfaction-driven
w = 1   : exit driven by low satisfaction only (load enters only through behaviour)
s0 is set so that baseline turnover stays close to 14% in every variant.
Run: python ablation.py   (needs pema_work_abl.py = pema_work.py with the mixed hazard)"""
import json, numpy as np
from sklearn.metrics import roc_auc_score
import pema_work_abl as m
VARIANTS = {"load-only (w=0)": dict(sat_w=0.0, sat_s0=0.30),
            "mixed (w=0.5)":   dict(sat_w=0.5, sat_s0=0.35),
            "satisfaction-only (w=1)": dict(sat_w=1.0, sat_s0=0.415)}
def one(P, seed):
    o = m.run({**P, "seed": seed}, record=True)
    tr, ed, T = o["trace"], o["exit_day"], o["P"]["T"]
    rng = np.random.default_rng(seed + 999); rows = []
    for _ in range(4000):
        i = rng.integers(len(ed)); t = rng.integers(30, T - 30)
        if np.isnan(tr["L"][i, t]): continue
        rows.append((tr["L"][i, t], tr["S"][i, t], tr["OCB"][i, t], int(ed[i] >= t and ed[i] < t + 30)))
    r = np.array(rows)
    if r[:, 3].sum() <= 5: return (np.nan,) * 3 + (m.turnover_rate(o),)
    return (roc_auc_score(r[:, 3], r[:, 0]), roc_auc_score(r[:, 3], -r[:, 1]),
            roc_auc_score(r[:, 3], -r[:, 2]), m.turnover_rate(o))
if __name__ == "__main__":
    out = {}
    for name, P in VARIANTS.items():
        a = np.array([one(P, s) for s in range(50)])
        out[name] = dict(auc_load=[a[:,0].mean(), a[:,0].std()], auc_satisfaction=[a[:,1].mean(), a[:,1].std()],
                         auc_ocb=[a[:,2].mean(), a[:,2].std()], turnover=float(a[:,3].mean()))
        print(name, {k: (np.round(v, 3).tolist()) for k, v in out[name].items()})
    json.dump(out, open("out/ablation.json", "w"), indent=1)
