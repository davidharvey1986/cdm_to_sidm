"""Mass- and redshift-dependent radial template, fitted to noise-reduced per-cluster SIDM responses.

Per cluster (CDM map with every sigma of its suite, |shift| <= 2): per pixel  delta_s = a + b s^p  over the suite's
sigmas (p = the suite's template exponent); b = response at sigma = 1 with the CDM run's shared noise removed (a).
Radial profile of b, then per radius bin r (OLS over training clusters):
    b(r) = A0(r) + AM(r) (log10 M200 - 14.5) + Az(r) (z - 0.2)
Prediction for any pair: delta(r) = sigma^p * b(r; M, z).
Tested on (i) the suite's held-out clusters and (ii) every pair of the other suite, against the template without
mass/z (A0 only, refitted the same way). Metrics on single pairs: core (r < 2 px) RMSE, median by mass bin; on
noise-reduced b: core RMSE / correlation.
usage: python3 mass_template.py            -> runs/mass_template/{summary.json, results.png}
"""
import os, json
import numpy as np, matplotlib
from scipy.stats import spearmanr
matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/runs/mass_template"; os.makedirs(OUT, exist_ok=True)
N = 96; yy, xx = np.indices((N, N)); R = np.hypot(yy - 47.5, xx - 47.5); RB = np.minimum(R.astype(int), 47)
CNT = np.bincount(RB.ravel(), minlength=48); CORE_W = np.zeros(48); CORE_W[:2] = [((R < 2) & (RB == k)).sum() for k in (0, 1)]
CORE_W /= CORE_W.sum(); M0, Z0 = 14.5, 0.2
MB = [14.0, 14.3, 14.5, 14.7, 15.0, 15.6]


def rprof(m):
    return np.stack([np.bincount(RB.ravel(), v.ravel(), 48) for v in m.reshape(-1, N, N)]).reshape(*m.shape[:-2], 48) / CNT


def load(name):
    f, p_dir, test = {"bahamas": ("pairs.npz", "two_scale_bahamas_nocnn", "test"),
                      "darkskies": ("darkskies_pairs.npz", "two_scale_darkskies_nocnn", "val")}[name]
    D = np.load(f"{HERE}/data/{f}", allow_pickle=True)
    x, y, xi, sig, sh, spl, z, M = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "split", "redshift", "log10_M200"))
    p = json.load(open(f"{HERE}/runs/{p_dir}/template.json"))["template"]["p"]
    ok = (np.abs(sh).max(1) <= 2) & (sig > 0); SG = np.unique(sig[ok])
    rows = {}
    for i in np.where(ok)[0]: rows.setdefault(xi[i], {})[float(sig[i])] = i
    cl = np.array([c for c, d in rows.items() if all(float(s) in d for s in SG)])
    X = x[cl, 2:98, 2:98].astype(np.float64)
    Dl = np.stack([y[[rows[c][float(s)] for c in cl], 2:98, 2:98] for s in SG], 1) - X[:, None]
    w = SG ** p; A_ = np.stack([np.ones(len(SG)), w], 1); coef = np.einsum("ks,nsij->nkij", np.linalg.pinv(A_), Dl)
    bprof = rprof(coef[:, 1])
    # single pairs (all, for testing)
    k = np.where(ok)[0]
    pair = dict(prof=rprof((y[k] - x[xi[k]])[:, 2:98, 2:98].astype(np.float64)), sigma=sig[k], M=M[xi[k]], z=z[xi[k]],
                test=spl[xi[k]] == test)
    return dict(name=name, p=p, SG=SG, bprof=bprof, M=M[cl], z=z[cl], test=spl[cl] == test, pair=pair)


def design(M, z, mass):
    return np.stack([np.ones_like(M), M - M0, z - Z0], 1) if mass else np.ones((len(M), 1))


def fit(S, mass):
    tr = ~S["test"]; Xd = design(S["M"][tr], S["z"][tr], mass)
    C = np.linalg.lstsq(Xd, S["bprof"][tr], rcond=None)[0]  # (k, 48)
    # bootstrap core coefficients
    rng = np.random.default_rng(0); bs = []
    idx = np.where(tr)[0]
    for _ in range(200):
        j = rng.choice(idx, len(idx)); bs.append(np.linalg.lstsq(design(S["M"][j], S["z"][j], mass), S["bprof"][j] @ CORE_W, rcond=None)[0])
    return dict(C=C, p=S["p"], mass=mass, core_coef=(C @ CORE_W).tolist(), core_coef_err=np.std(bs, 0).tolist())


def predict_prof(T, sigma, M, z):
    return (sigma[:, None] ** T["p"]) * (design(M, z, T["mass"]) @ T["C"])


def score(T, S, pairs_mask):
    P = S["pair"]; m = pairs_mask
    pred = predict_prof(T, P["sigma"][m], P["M"][m], P["z"][m]) @ CORE_W; true = P["prof"][m] @ CORE_W
    out = {}
    for s in np.unique(P["sigma"][m]):
        k = P["sigma"][m] == s; Mk = P["M"][m][k]
        bins = lambda v: [float(np.median(v[(Mk >= lo) & (Mk < hi)])) if ((Mk >= lo) & (Mk < hi)).sum() > 8 else None for lo, hi in zip(MB[:-1], MB[1:])]
        out[str(float(s))] = dict(n=int(k.sum()), rmse=float(np.sqrt(np.mean((pred[k] - true[k]) ** 2))),
                                  bias=float(np.median(pred[k]) - np.median(true[k])), corr=float(np.corrcoef(pred[k], true[k])[0, 1]) if pred[k].std() > 0 else 0.0,
                                  true_by_M=bins(true[k]), pred_by_M=bins(pred[k]))
    return out


def score_b(T, S):
    """noise-reduced b (core) on held-out clusters of the same suite"""
    te = S["test"]; pb = (design(S["M"][te], S["z"][te], T["mass"]) @ T["C"]) @ CORE_W; tb = S["bprof"][te] @ CORE_W
    return dict(rmse=float(np.sqrt(np.mean((pb - tb) ** 2))), corr=float(np.corrcoef(pb, tb)[0, 1]) if pb.std() > 0 else 0.0, n=int(te.sum()))


suites = {n: load(n) for n in ("bahamas", "darkskies")}
res = {}
for tr_name, S in suites.items():
    ot = suites["darkskies" if tr_name == "bahamas" else "bahamas"]
    print(f"\n######## trained on {tr_name}: {(~S['test']).sum()} train clusters (complete sigma sets {S['SG'].tolist()}), p = {S['p']:.3f}")
    R_ = {}
    for mass in (False, True):
        T = fit(S, mass); tag = "mass+z" if mass else "no mass"
        cc, ce = np.array(T["core_coef"]), np.array(T["core_coef_err"])
        print(f"  [{tag}] core b at sigma=1: " + ", ".join(f"{nm} {v:+.4f}±{e:.4f}" for nm, v, e in zip(("A0", "dB/dlogM", "dB/dz")[:len(cc)], cc, ce)))
        R_[tag] = dict(core_coef=T["core_coef"], core_coef_err=T["core_coef_err"], b_test=score_b(T, S),
                       in_suite=score(T, S, S["pair"]["test"]), cross_suite=score(T, ot, np.ones(len(ot["pair"]["sigma"]), bool)),
                       C=T["C"].tolist())
        print(f"     noise-reduced b, held-out clusters: core RMSE {R_[tag]['b_test']['rmse']:.4f} corr {R_[tag]['b_test']['corr']:+.2f}")
    for where in ("in_suite", "cross_suite"):
        print(f"  {where.replace('_', '-')} single pairs, core (r<2): RMSE no-mass -> mass+z | corr | median by log M bin {MB}")
        for s in R_["no mass"][where]:
            a0, a1 = R_["no mass"][where][s], R_["mass+z"][where][s]
            f = lambda v: "[" + " ".join("  -  " if u is None else f"{u:+.3f}" for u in v) + "]"
            print(f"   sigma {float(s):<5} n {a0['n']:5d}: {a0['rmse']:.4f} -> {a1['rmse']:.4f} | corr {a1['corr']:+.2f} | true {f(a1['true_by_M'])} mass+z {f(a1['pred_by_M'])} no-mass {a0['pred_by_M'][2]:+.3f}")
    res[tr_name] = R_
json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)

# figure: core b vs mass for both suites, with each suite's fitted mass+z line
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
cols = {"bahamas": "#3a6fb0", "darkskies": "#c0392b"}
for nm, S in suites.items():
    bc = S["bprof"] @ CORE_W; Mg = np.linspace(14.1, 15.3, 20)
    edges = np.linspace(14.0, 15.4, 8); mid = 0.5 * (edges[1:] + edges[:-1])
    med = [np.median(bc[(S["M"] >= lo) & (S["M"] < hi)]) if ((S["M"] >= lo) & (S["M"] < hi)).sum() > 10 else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
    ax[0].plot(mid, med, "o", color=cols[nm], label=f"{nm}: median b (core)")
    cc = res[nm]["mass+z"]["core_coef"]; ax[0].plot(Mg, cc[0] + cc[1] * (Mg - M0), color=cols[nm], lw=1.5, label=f"{nm} fit (z={Z0})")
    for z_ in np.unique(S["z"]):
        k = S["z"] == z_
        ax[1].plot(z_, np.median(bc[k]), "s" if nm == "bahamas" else "^", color=cols[nm], ms=8)
    C = np.array(res[nm]["mass+z"]["C"]); rr = np.arange(48) + 0.5; rr[0] = 0.35
    for (Mx, ls) in ((14.2, ":"), (14.5, "-"), (15.0, "--")):
        ax[2].plot(rr, C[0] + C[1] * (Mx - M0), color=cols[nm], ls=ls, lw=1.4, label=f"{nm} logM={Mx}")
ax[0].set_xlabel("log10 M200"); ax[0].set_ylabel("core response b at σ=1 [dex]"); ax[0].legend(fontsize=7); ax[0].set_title("mass dependence")
ax[1].set_xlabel("z"); ax[1].set_title("redshift dependence (median core b; ■ BAHAMAS ▲ darkskies)")
ax[2].set_xscale("log"); ax[2].axhline(0, color="0.5", lw=0.7); ax[2].set_xlabel("r [px]"); ax[2].set_ylabel("b(r) at σ=1, z=0.2")
ax[2].legend(fontsize=6, ncol=2); ax[2].set_title("fitted radial templates")
fig.savefig(f"{OUT}/results.png", dpi=105); print("wrote", OUT)
