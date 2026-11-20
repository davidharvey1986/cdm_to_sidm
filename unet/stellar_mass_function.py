"""Projected stellar mass function of the cluster galaxies from the stellar maps (BAHAMAS test split): CDM, true SIDM,
and the stars-flow draws with the true SIDM DM ("oracle", runs/stars_true) and with the DM-flow draws ("chain", runs/stars_hyb).

Maps: log10(Sigma_star + 1e11) (pairs_stars.npz), linear Sigma_star = 10^l - 1e11, 96x96 px of 20 kpc.
Satellites: peaks of Sigma_star - median_filter(Sigma_star, 9 px) (local max in 5x5 px) at 80-920 kpc; stellar mass
  M*_40 = residual within 2 px (40 kpc) x pixel area. A projected aperture mass (smoothing 30 kpc, |los| < 5 pMpc).
Central galaxy: M*(<50 kpc) of the full map (no background subtraction).
Resolution: one BAHAMAS star particle ~1e9 Msun, so M*_40 < ~1e10 is a few particles (shaded in the figure).
Figure figures/stellar_mass_function.png, rows sigma: cumulative N(>M*) of satellites per map; ratio to CDM; radial number
density of satellites with M* > 10^10.3; distribution of the central-galaxy mass change log M*_SIDM/M*_CDM.
usage: python stellar_mass_function.py
"""
import json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, maximum_filter, convolve
import dm_common as C, two_scale as T2
MED, RMIN, RMAX, MMIN, M_N, MRES, NDRAW, A_PX = 9, 4, 46, 10 ** 9.3, 10 ** 10.3, 1e10, 4, 0.02 ** 2
DISK = (np.hypot(*np.mgrid[-2:3, -2:3]) <= 2).astype(np.float64); FIELD = (T2.R >= RMIN) & (T2.R < RMAX); CEN = T2.R < 2.5
MB = np.logspace(9.3, 11.6, 24); RB = np.arange(RMIN, RMAX + 1, 3); RBk = (RB[:-1] + RB[1:]) / 2 * 20
AREA = np.array([((T2.R >= a) & (T2.R < b)).sum() for a, b in zip(RB[:-1], RB[1:])]) * A_PX
lin = lambda l: np.maximum(10 ** l.astype(np.float64) - 1e11, 0)


def measure(L):
    """L (n, 96, 96) -> N(>M*) per map (n, len(MB)), n(R) for M* > M_N (n, len(RBk)), central M*(<50 kpc) (n,)"""
    N, nR, Mc = [], [], []
    for l in L:
        S = lin(l); res = S - median_filter(S, MED, mode="nearest"); M = convolve(res, DISK, mode="nearest") * A_PX
        pk = (maximum_filter(res, 5) == res) & FIELD & (M > MMIN); m, r = M[pk], T2.R[pk]
        N.append((m[:, None] > MB[None]).sum(0)); nR.append(np.histogram(r[m > M_N], RB)[0] / AREA); Mc.append(S[CEN].sum() * A_PX)
    return np.array(N), np.array(nR), np.array(Mc)


S, _ = C.load(); te = S["test"]; sig = te["sigma"]; D = np.load(f"{C.HERE}/data/pairs_stars.npz", allow_pickle=True)
xs = D["x"][D["xi"][te["k"]]][:, 2:98, 2:98]; ys = D["y"][te["k"]][:, 2:98, 2:98]
R = {"CDM": measure(xs), "true": measure(ys)}; print("CDM, true done", flush=True)
for nm, f in (("oracle", "stars_true"), ("chain", "stars_hyb")):
    X = np.load(f"{C.HERE}/runs/{f}/draws.npz")["draws"][:NDRAW].astype(np.float32)
    parts = [measure(X[d]) for d in range(len(X))]
    R[nm] = (np.mean([p[0] for p in parts], 0), np.mean([p[1] for p in parts], 0), np.stack([p[2] for p in parts])); print(nm, "done", flush=True)
COL = {"CDM": "0.55", "true": "k", "oracle": "#2a9d8f", "chain": "#d1495b"}
LAB = {"CDM": "CDM", "true": "true SIDM", "oracle": "flow, true SIDM DM", "chain": "flow, DM-flow draw"}
mse = lambda a: (a.mean(0), a.std(0) / np.sqrt(len(a)))
fig, ax = plt.subplots(3, 4, figsize=(25, 15.5), constrained_layout=True); summ = {}
for i, s in enumerate(C.SIG):
    m = sig == s; A = ax[i]; summ[str(s)] = {}; NC = R["CDM"][0][m].mean(0); nC = R["CDM"][1][m].mean(0)
    for nm, (N, nR, Mc) in R.items():
        cl, lw, ls = COL[nm], (2.6 if nm in ("CDM", "true") else 1.8), ("-" if nm in ("CDM", "true") else "--")
        mu, se = mse(N[m]); A[0].plot(MB, mu, color=cl, lw=lw, ls=ls, label=LAB[nm]); A[0].fill_between(MB, mu - se, mu + se, color=cl, alpha=0.15, lw=0)
        mu, se = mse(nR[m]); A[2].plot(RBk, mu, color=cl, lw=lw, ls=ls, label=LAB[nm]); A[2].fill_between(RBk, mu - se, mu + se, color=cl, alpha=0.15, lw=0)
        if nm != "CDM":
            A[1].plot(MB, N[m].mean(0) / NC, color=cl, lw=lw, ls=ls, label=LAB[nm])
            dc = np.log10(Mc[..., m] / R["CDM"][2][m]).ravel()
            A[3].hist(dc, np.linspace(-0.6, 0.6, 61), histtype="step", density=True, color=cl, lw=lw, ls=ls, label=f"{LAB[nm]} (med {np.median(dc):+.3f}, sd {dc.std():.3f})")
        j10, j11 = int(np.argmin(np.abs(MB - 10 ** 10.3))), int(np.argmin(np.abs(MB - 1e11)))
        summ[str(s)][nm] = dict(N_gt_10p3=float(N[m, j10].mean()), N_gt_11=float(N[m, j11].mean()), n_in=float(nR[m, :3].mean()), n_out=float(nR[m, -3:].mean()))
    for a in A[:2]: a.axvspan(MB[0], MRES, color="0.9", zorder=0); a.set_xscale("log"); a.set_xlabel("M*_40 [Msun]")
    A[0].set_yscale("log"); A[0].set_ylim(0.02, None); A[0].set_ylabel(f"SIDM{s:g} ({m.sum()} test maps)\nsatellites N(> M*) per map", fontsize=12)
    A[1].axhline(1, color="0.6", lw=0.6); A[1].set_ylim(0.5, 1.5); A[1].set_ylabel("N(>M*) / N_CDM(>M*)")
    A[2].set_yscale("log"); A[2].set_xlabel("R [kpc]"); A[2].set_ylabel("n(R) [pMpc⁻²], M* > 10^10.3")
    A[3].set_xlabel("log10 M*_SIDM / M*_CDM, central galaxy (< 50 kpc)")
    for a in A: a.legend(fontsize=8)
for j, t in enumerate(["satellite stellar mass function (80–920 kpc)", "relative to CDM (grey band: < ~10 star particles)", "satellite radial number density", "central-galaxy stellar-mass change"]):
    ax[0, j].set_title(t, fontsize=13)
fig.suptitle("Projected stellar mass function, BAHAMAS test split: CDM, true SIDM and the stars flow (true or flow-drawn SIDM dark matter)", fontsize=15)
fig.savefig(f"{C.HERE}/figures/stellar_mass_function.png", dpi=62); json.dump(summ, open(f"{C.HERE}/runs/stellar_mass_function_summary.json", "w"), indent=1)
for s, d in summ.items():
    for nm, r in d.items(): print(f"SIDM{s} {nm:7s} " + " ".join(f"{k}={v:.3f}" for k, v in r.items()))
print("wrote figures/stellar_mass_function.png")
