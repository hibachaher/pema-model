"""Experiments for PEMA-Work. Writes results to out/*.npz / json."""
import json, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.stats import qmc
from sklearn.metrics import roc_auc_score
from pema_work import run, turnover_rate, shock_experiment, DEFAULTS

OUT = "out"

def baseline_one(seed):
    o = run({"seed": seed}, record=True)
    tr = o["trace"]; ed = o["exit_day"]; ec = o["exit_cause"]
    T = o["P"]["T"]
    # --- H3: predictive AUC (exit within next 30 days) from L(t) vs S(t)
    rng = np.random.default_rng(seed + 999)
    rows = []
    for _ in range(4000):
        i = rng.integers(len(ed)); t = rng.integers(30, T - 30)
        if np.isnan(tr["L"][i, t]):
            continue
        y = int(ed[i] >= t and ed[i] < t + 30)
        rows.append((tr["L"][i, t], tr["S"][i, t], tr["OCB"][i, t], y))
    rows = np.array(rows)
    auc_L = auc_S = auc_O = np.nan
    if rows[:, 3].sum() > 5:
        auc_L = roc_auc_score(rows[:, 3], rows[:, 0])
        auc_S = roc_auc_score(rows[:, 3], -rows[:, 1])
        auc_O = roc_auc_score(rows[:, 3], -rows[:, 2])
    # --- H4: event-time ordering for overload exits
    k = 7
    def roll(x):
        c = np.cumsum(np.nan_to_num(x, nan=0.0)); r = np.full_like(x, np.nan)
        r[k - 1:] = (c[k - 1:] - np.concatenate([[0], c[:-k]])) / k
        return r
    order = []; aligned = []
    W = 400
    for i in np.where((ed >= 0) & (ec == 1))[0]:
        o_ = roll(tr["OCB"][i, :ed[i] + 1]); c_ = roll(tr["CWB"][i, :ed[i] + 1])
        base = np.nanmean(tr["OCB"][i, :60]) if ed[i] > 60 else np.nan
        thr = 0.6 * base
        t_o = np.argmax(o_ < thr) if (o_ < thr).any() else -1
        t_c = np.argmax(c_ > 0.40) if (c_ > 0.40).any() else -1
        order.append((t_o, t_c, ed[i]))
        if ed[i] >= W:
            seg = np.stack([tr[v][i, ed[i] - W:ed[i] + 1] for v in ("L", "S", "OCB", "CWB")])
            aligned.append(seg)
    aligned = np.array(aligned) if aligned else np.zeros((0, 4, W + 1))
    return dict(seed=seed, turnover=turnover_rate(o), spells=o["spells"], mean_ocb=o["mean_ocb"],
                mean_cwb=o["mean_cwb"], auc=(auc_L, auc_S, auc_O), order=np.array(order),
                aligned_mean=aligned.mean(0) if len(aligned) else None, n_aligned=len(aligned),
                monthly=o["monthly_exits"])

SA_NAMES = ["kappa", "p_neg", "alpha", "rho0", "omega_scale", "theta_mu", "eta"]
SA_BOUNDS = [[0.0016, 0.0030], [0.24, 0.36], [0.08, 0.25], [0.0006, 0.0018],
             [0.5, 1.5], [0.85, 1.15], [0.04, 0.12]]

def sa_one(args):
    k, x = args
    P = dict(zip(SA_NAMES, x)); P.update(N=150, T=730, seed=10000 + k)
    o = run(P)
    return turnover_rate(o, years_from=0.5), o["mean_ocb"]

SCEN = {
    "Baseline": {},
    "Stressor reduction (-20%)": {"p_neg": 0.24},
    "Universal recovery programme": {"program_rho": 0.0004},
    "Late support (CWB onset)": {"target": "late"},
    "Early support (OCB drop)": {"target": "early"},
}

def scen_one(args):
    name, seed = args
    P = {**SCEN[name], "seed": 5000 + seed}
    o = run(P)
    tdays = o["treat_days"] if name != "Universal recovery programme" else o["P"]["N"] * o["P"]["T"]
    return name, turnover_rate(o), o["mean_ocb"], o["mean_cwb"], tdays

if __name__ == "__main__":
    what = sys.argv[1]
    t0 = time.time()
    with Pool(2) as pool:
        if what == "baseline":
            res = pool.map(baseline_one, range(50))
            np.save(f"{OUT}/baseline.npy", np.array(res, dtype=object), allow_pickle=True)
        elif what == "sa":
            d = len(SA_NAMES); n = 128
            sob = qmc.Sobol(2 * d, seed=7).random(n)
            lo = np.array([b[0] for b in SA_BOUNDS]); hi = np.array([b[1] for b in SA_BOUNDS])
            A = lo + sob[:, :d] * (hi - lo); B = lo + sob[:, d:] * (hi - lo)
            X = [A, B] + [np.where(np.arange(d) == j, B, A) for j in range(d)]
            X = np.vstack(X)
            Y = np.array(pool.map(sa_one, list(enumerate(X))))
            np.savez(f"{OUT}/sa.npz", X=X, Y=Y, n=n, d=d)
        elif what == "scen":
            jobs = [(nm, s) for nm in SCEN for s in range(30)]
            res = pool.map(scen_one, jobs)
            json.dump(res, open(f"{OUT}/scen.json", "w"))
        elif what == "shock":
            r = shock_experiment()
            np.savez(f"{OUT}/shock.npz", psi=r["psi"], drop=r["drop"], rec=r["rec"],
                     traj=r["traj"][::20], S_base=r["S_base"])
    print(what, "done in", round(time.time() - t0), "s")
