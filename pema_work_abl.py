"""
PEMA-Work: Personality-EMotion Agent model of workplace allostatic load,
OCB->CWB erosion and turnover. Python/NumPy implementation of the
model described in the manuscript.

Time step = one working day. One run = one organisation of N positions;
leavers are replaced by new hires so that headcount stays constant.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import truncnorm

TRAITS = ["O", "C", "E", "A", "N"]
EMOTIONS = ["Joy", "Distress", "Happy-for", "Pity", "Gloating", "Resentment",
            "Hope", "Fear", "Satisfaction", "Fears-confirmed", "Relief",
            "Disappointment", "Pride", "Shame", "Admiration", "Reproach",
            "Gratification", "Remorse", "Gratitude", "Anger"]
NEG = ["Distress", "Pity", "Resentment", "Fear", "Fears-confirmed",
       "Disappointment", "Shame", "Reproach", "Remorse", "Anger"]
POS = [e for e in EMOTIONS if e not in NEG]

# ---------------------------------------------------------------------------
# Global sensitivity matrix Omega (traits x emotions), entries in [-1, 1].
# Signs follow the trait-affect literature (see Table 2 of the paper);
# magnitudes are priors and are varied jointly in the Sobol analysis.
# ---------------------------------------------------------------------------
_OM = {
    #            O      C      E      A      N
    "Joy":            (0.20, 0.10, 0.60, 0.20, -0.30),
    "Distress":       (-0.20, -0.35, -0.10, 0.00, 0.91),
    "Happy-for":      (0.20, 0.00, 0.30, 0.55, -0.10),
    "Pity":           (0.20, 0.00, 0.00, 0.50, 0.30),
    "Gloating":       (0.00, -0.10, 0.10, -0.60, 0.40),
    "Resentment":     (-0.40, 0.00, 0.00, -0.50, 0.70),
    "Hope":           (0.30, 0.20, 0.40, 0.10, -0.30),
    "Fear":           (-0.30, 0.00, -0.10, 0.00, 0.85),
    "Satisfaction":   (0.10, 0.40, 0.30, 0.10, -0.20),
    "Fears-confirmed":(-0.20, -0.10, 0.00, 0.00, 0.80),
    "Relief":         (0.10, 0.10, 0.20, 0.10, 0.20),
    "Disappointment": (-0.20, 0.20, -0.10, 0.00, 0.70),
    "Pride":          (0.10, 0.45, 0.35, 0.00, -0.20),
    "Shame":          (-0.51, 0.10, -0.20, 0.10, 0.75),
    "Admiration":     (0.30, 0.10, 0.20, 0.50, 0.00),
    "Reproach":       (-0.10, 0.20, 0.00, -0.55, 0.50),
    "Gratification":  (0.10, 0.45, 0.30, 0.00, -0.20),
    "Remorse":        (0.00, -0.30, -0.10, 0.20, 0.60),
    "Gratitude":      (0.20, 0.10, 0.30, 0.55, 0.00),
    "Anger":          (-0.10, -0.20, 0.10, -0.55, 0.80),
}
OMEGA = np.array([_OM[e] for e in EMOTIONS]).T          # shape (5, 20)
IDX = {e: i for i, e in enumerate(EMOTIONS)}

# OCC event types -> emotions they can elicit (with OCC appraisal target)
EVENT_TYPES = {
    # negative: (probability share, emotions)
    "neg": [
        (0.29, ["Distress", "Disappointment", "Fear"]),          # technical / operational difficulty
        (0.19, ["Distress", "Resentment", "Disappointment"]),    # below-expectation pay decision
        (0.22, ["Reproach", "Anger", "Distress"]),               # blameworthy action of supervisor / peer
        (0.15, ["Shame", "Remorse"]),                            # own failure
        (0.15, ["Pity", "Fears-confirmed", "Distress"]),         # bad event affecting colleague / team
    ],
    "pos": [
        (0.25, ["Joy", "Pride", "Gratification"]),              # promotion / recognition
        (0.20, ["Joy", "Satisfaction", "Pride"]),               # task success
        (0.30, ["Admiration", "Gratitude", "Joy"]),             # praise / help from peers
        (0.15, ["Happy-for", "Joy"]),                           # good event for colleague
        (0.10, ["Hope", "Relief"]),                             # positive outlook / threat removed
    ],
}

DEFAULTS = dict(
    N=300, T=1000,
    p_neg=0.30, p_pos=0.25,      # daily probability of a negative / positive appraised event
    alpha=0.15,                  # dissipation scale: delta = alpha/(1+N)
    eta=0.08,                    # weight of net appraisal on job satisfaction
    kappa=0.0023,                # conversion of lingering negative affect into allostatic load
    rho0=0.0012,                 # allostatic recovery fuelled by positive affect
    omega_scale=1.0,             # multiplies Omega (sensitivity analysis)
    theta_mu=1.0, theta_sd=0.10, # resignation tolerance theta_turnover
    h0=0.0001,                   # stress-unrelated daily exit hazard (~2.5%/year)
    h_max=0.08,                  # max daily exit hazard under allostatic overload
    erosion=0.45, critical=0.75, # phase thresholds on AL/theta
    hire_C_mu=0.5,
    # interventions
    program_rho=0.0,             # universal recovery programme (added AL recovery per day)
    target="none",               # "none" | "early" (OCB drop) | "late" (CWB onset)
    target_rho=0.006, target_days=60,
    sat_w=0.0, sat_s0=0.30, sat_width=0.15,   # ablation only: share of hazard driven by low satisfaction
    seed=0,
)

# ---------------------------------------------------------------------------
# Mamdani fuzzy inference for OCB / CWB propensities (precomputed on a grid)
# ---------------------------------------------------------------------------
def _trap(x, a, b, c, d):
    return np.clip(np.minimum((x - a) / (b - a + 1e-12), (d - x) / (d - c + 1e-12)), 0, 1)

def _tri(x, a, b, c):
    return _trap(x, a, b, b, c)

def build_fuzzy(erosion=0.45, critical=0.75):
    s = np.linspace(0, 1, 41)
    l = np.linspace(0, 1.3, 53)
    a = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 201)
    S, L, A = np.meshgrid(s, l, a, indexing="ij")
    # input membership functions
    sL, sM, sH = _trap(S, -1, 0, 0.3, 0.5), _tri(S, 0.3, 0.55, 0.8), _trap(S, 0.6, 0.8, 1, 2)
    e, c = erosion, critical
    lH = _trap(L, -1, 0, e - 0.08, e + 0.04)             # healthy
    lE = _trap(L, e - 0.08, e + 0.04, c - 0.06, c + 0.04)  # eroding
    lC = _trap(L, c - 0.06, c + 0.04, 2, 3)              # critical
    aL, aH = _trap(A, -1, 0, 0.35, 0.65), _trap(A, 0.35, 0.65, 1, 2)
    # output sets
    oL, oM, oH = _trap(y, -1, 0, 0.15, 0.4), _tri(y, 0.25, 0.5, 0.75), _trap(y, 0.6, 0.85, 1, 2)
    mn = np.minimum
    ocb_rules = [
        (mn(lH, sH), oH), (mn(mn(lH, sM), aH), oH), (mn(mn(lH, sM), aL), oM),
        (mn(lE, aH), oM), (mn(lE, aL), oL), (lC, oL), (sL, oL),
    ]
    cwb_rules = [
        (mn(lC, aL), oH), (mn(lC, aH), oM), (mn(mn(lE, sL), aL), oM),
        (lH, oL), (sH, oL), (mn(lE, aH), oL),
    ]
    def infer(rules):
        agg = np.zeros(S.shape + (y.size,))
        for w, out in rules:
            agg = np.maximum(agg, np.minimum(w[..., None], out))
        num = (agg * y).sum(-1); den = agg.sum(-1) + 1e-12
        return num / den
    grid = (s, l, a)
    return (RegularGridInterpolator(grid, infer(ocb_rules), bounds_error=False, fill_value=None),
            RegularGridInterpolator(grid, infer(cwb_rules), bounds_error=False, fill_value=None))

_FUZZY_CACHE = {}
def fuzzy(erosion, critical):
    k = (round(erosion, 4), round(critical, 4))
    if k not in _FUZZY_CACHE:
        _FUZZY_CACHE[k] = build_fuzzy(erosion, critical)
    return _FUZZY_CACHE[k]

# ---------------------------------------------------------------------------
def tn(rng, mu, sd, size):
    a, b = (0 - mu) / sd, (1 - mu) / sd
    return truncnorm.rvs(a, b, loc=mu, scale=sd, size=size, random_state=rng)

def new_agents(rng, n, P):
    psi = np.column_stack([tn(rng, 0.5, 0.15, n), tn(rng, P["hire_C_mu"], 0.15, n),
                           tn(rng, 0.5, 0.15, n), tn(rng, 0.5, 0.15, n), tn(rng, 0.5, 0.15, n)])
    return psi

def gains(psi, scale):
    # personality gain per emotion, centred on 1 for an average profile
    return np.clip(1 + scale * (psi - 0.5) @ OMEGA, 0.05, 2.0)       # (n, 20)

def run(P=None, record=False, rec_every=1):
    P = {**DEFAULTS, **(P or {})}
    rng = np.random.default_rng(P["seed"])
    N, T = P["N"], P["T"]
    ocb_f, cwb_f = fuzzy(P["erosion"], P["critical"])
    psi = new_agents(rng, N, P)
    G = gains(psi, P["omega_scale"])
    O, C, E, A, Nn = psi.T
    def init_state(idx):
        st = {}
        st["S_base"] = np.clip(0.62 + 0.20 * (E[idx] - 0.5) - 0.25 * (Nn[idx] - 0.5), 0.2, 0.95)
        return st
    S_base = np.clip(0.62 + 0.20 * (E - 0.5) - 0.25 * (Nn - 0.5), 0.2, 0.95)
    S = S_base.copy()
    NA = np.zeros(N); PA = np.zeros(N); AL = np.zeros(N)
    IES = np.zeros((N, 20))
    theta = np.clip(rng.normal(P["theta_mu"], P["theta_sd"], N), 0.5, 1.6)
    start = np.zeros(N, dtype=int)
    treat_left = np.zeros(N, dtype=int); treated_flag = np.zeros(N, bool)
    ocb_base = None
    spells = []           # (traits..., tenure, censored, cause)
    treat_days = 0
    neg_types, pos_types = EVENT_TYPES["neg"], EVENT_TYPES["pos"]
    neg_p = np.array([t[0] for t in neg_types]); neg_p /= neg_p.sum()
    pos_p = np.array([t[0] for t in pos_types]); pos_p /= pos_p.sum()
    neg_masks = np.zeros((len(neg_types), 20)); pos_masks = np.zeros((len(pos_types), 20))
    for k, (_, em) in enumerate(neg_types):
        neg_masks[k, [IDX[e] for e in em]] = 1
    for k, (_, em) in enumerate(pos_types):
        pos_masks[k, [IDX[e] for e in em]] = 1
    negcol = np.array([e in NEG for e in EMOTIONS])
    rec = {k: [] for k in ["S", "AL", "L", "OCB", "CWB", "NA"]} if record else None
    rec_ids = None
    trace = {}             # per-agent trajectories for event-time analysis (first cohort)
    if record:
        trace = {k: np.full((N, T), np.nan, dtype=np.float32) for k in ["S", "L", "OCB", "CWB"]}
        alive_orig = np.ones(N, bool)
        exit_day = np.full(N, -1); exit_cause = np.zeros(N, int)
    monthly_exits = np.zeros(T // 30 + 1)
    ocb_sum = 0.0; cwb_sum = 0.0
    for t in range(T):
        delta = P["alpha"] / (1 + Nn)
        # --- perception: daily appraised events
        neg = rng.random(N) < P["p_neg"]
        pos = rng.random(N) < P["p_pos"]
        u_neg = rng.uniform(0.2, 1.0, N) * neg
        u_pos = rng.uniform(0.2, 1.0, N) * pos
        kn = rng.choice(len(neg_types), N, p=neg_p)
        kp = rng.choice(len(pos_types), N, p=pos_p)
        # --- appraisal: intensity = potency x personality gain (OCC + Omega)
        new = (u_neg[:, None] * neg_masks[kn] + u_pos[:, None] * pos_masks[kp]) * G
        new = np.clip(new, 0, 1)
        IES = np.clip(IES * np.exp(-delta)[:, None] + new, 0, 1)
        nm = neg_masks[kn]; pm = pos_masks[kp]
        I_neg = (new * nm).sum(1) / np.maximum(nm.sum(1), 1) * neg
        I_pos = (new * pm).sum(1) / np.maximum(pm.sum(1), 1) * pos
        # --- state update
        NA = NA * np.exp(-delta) + I_neg                      # lingering negative affect (affective inertia)
        PA = PA * np.exp(-delta) + I_pos
        S = np.clip(S_base + (S - S_base) * np.exp(-delta) + P["eta"] * (I_pos - I_neg), 0, 1)
        rho = P["rho0"] * PA + P["program_rho"]
        tr = treat_left > 0
        rho = rho + P["target_rho"] * tr
        treat_days += tr.sum()
        treat_left = np.maximum(treat_left - 1, 0)
        AL = np.maximum(0, AL + P["kappa"] * (1 - C) * NA - rho)
        L = AL / theta
        # --- behaviour (fuzzy)
        pts = np.column_stack([S, np.minimum(L, 1.3), A])
        ocb = np.clip(ocb_f(pts), 0, 1); cwb = np.clip(cwb_f(pts), 0, 1)
        ocb_sum += ocb.mean(); cwb_sum += cwb.mean()
        # --- targeted support triggers
        if P["target"] != "none":
            if P["target"] == "early":
                trig = (L >= P["erosion"]) & (ocb < 0.45)
            else:
                trig = cwb > 0.5
            trig &= ~treated_flag
            treat_left[trig] = P["target_days"]; treated_flag[trig] = True
        # --- turnover: fuzzy overload hazard + baseline
        mu_over = np.clip((L - 0.92) / (1.08 - 0.92), 0, 1)
        mu_sat = np.clip((P["sat_s0"] - S) / P["sat_width"], 0, 1)
        mu_mix = (1 - P["sat_w"]) * mu_over + P["sat_w"] * mu_sat
        h = P["h0"] + P["h_max"] * mu_mix
        quit = rng.random(N) < h
        if record:
            idx = np.where(alive_orig)[0]
            for k, v in (("S", S), ("L", L), ("OCB", ocb), ("CWB", cwb)):
                trace[k][idx, t] = v[idx]
            q = quit & alive_orig
            exit_day[q] = t; exit_cause[q] = np.where(mu_over[q] > 0, 1, 0)
            alive_orig &= ~quit
        if quit.any():
            qi = np.where(quit)[0]
            monthly_exits[t // 30] += qi.size
            for i in qi:
                spells.append((*psi[i], t - start[i], 0, int(mu_over[i] > 0)))
            m = qi.size
            psi[qi] = new_agents(rng, m, P)
            O, C, E, A, Nn = psi.T
            G[qi] = gains(psi[qi], P["omega_scale"])
            S_base[qi] = np.clip(0.62 + 0.20 * (E[qi] - 0.5) - 0.25 * (Nn[qi] - 0.5), 0.2, 0.95)
            S[qi] = S_base[qi]; NA[qi] = 0; PA[qi] = 0; AL[qi] = 0; IES[qi] = 0
            theta[qi] = np.clip(rng.normal(P["theta_mu"], P["theta_sd"], m), 0.5, 1.6)
            start[qi] = t; treat_left[qi] = 0; treated_flag[qi] = False
    for i in range(N):
        spells.append((*psi[i], T - start[i], 1, 0))
    out = dict(spells=np.array(spells, dtype=float), monthly_exits=monthly_exits,
               mean_ocb=ocb_sum / T, mean_cwb=cwb_sum / T, treat_days=int(treat_days), P=P)
    if record:
        out.update(trace=trace, exit_day=exit_day, exit_cause=exit_cause, psi0=None)
    return out

def turnover_rate(out, years_from=1.0):
    """Annual turnover rate (exits / headcount / year) after a burn-in."""
    P = out["P"]; me = out["monthly_exits"]
    m0 = int(years_from * 12)
    months = (P["T"] // 30) - m0
    if months <= 0:
        return np.nan
    return me[m0:m0 + months].sum() / P["N"] / (months / 12)

def shock_experiment(n=2000, shock=1.0, t_shock=100, T=400, seed=1, alpha=0.15, eta=0.08):
    """H1: recovery of job satisfaction after a single acute negative event, no other events."""
    rng = np.random.default_rng(seed)
    P = {**DEFAULTS}
    psi = new_agents(rng, n, P)
    Nn = psi[:, 4]; E = psi[:, 2]
    G = gains(psi, 1.0)
    S_base = np.clip(0.62 + 0.20 * (E - 0.5) - 0.25 * (Nn - 0.5), 0.2, 0.95)
    S = S_base.copy()
    delta = alpha / (1 + Nn)
    mask = np.zeros(20); mask[[IDX["Distress"], IDX["Disappointment"], IDX["Fear"]]] = 1
    I = np.clip(shock * mask * G, 0, 1).sum(1) / 3
    traj = np.zeros((n, T))
    for t in range(T):
        S = S_base + (S - S_base) * np.exp(-delta) - (eta * I if t == t_shock else 0)
        traj[:, t] = S
    drop = S_base - traj[:, t_shock]
    rec = np.full(n, np.nan)
    after = traj[:, t_shock:]
    ok = (S_base[:, None] - after) <= 0.05 * drop[:, None]
    first = ok.argmax(1).astype(float); first[~ok.any(1)] = np.nan
    return dict(psi=psi, drop=drop, rec=first, traj=traj, S_base=S_base)
