"""Projected subhalo mass function and radial number density from the DM maps (BAHAMAS test split), measured the same
way on CDM, true SIDM and the flow draws (the draws have no halo catalogue).

Substructure finder (on linear Sigma, maps smoothed 1.5 px as in pairs_dm):
  residual = Sigma - median_filter(Sigma, MED px)      (host model follows the host's ellipticity, robust to the peaks)
  peaks    = local maxima of the residual in 5x5 px, outside the central RMIN px and inside RMAX px
  mass     = aperture mass M_40 = sum of the residual within 2 px (40 kpc) of the peak x pixel area (Msun);
             peaks with M_40 > MMIN kept. M_40 is a projected aperture mass, not a bound mass (smoothing spreads
             a subhalo beyond 40 kpc; |los| < 5 pMpc adds line-of-sight structures); compare like with like.
Resolution: BAHAMAS m_DM = 5.5e9 Msun, so M_40 ~ 1e11 is ~20 particles.
Figure figures/subhalo_maps.png, rows sigma: cumulative N(>M_40) per map (all R; R < 300 kpc), its ratio to CDM, radial
number density n(R) for M_40 > M_N and its ratio to CDM. Bands: error on the mean over maps.
Models: CDM, true SIDM, and runs/dm_<name>/draws.npz for the names given (first NDRAW draws).
usage: python subhalo_maps.py flow_hyb flow_hyb_mc
"""
import sys, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, maximum_filter, convolve
import dm_common as C, two_scale as T2
MED, RMIN, RMAX, MMIN, M_N, NDRAW = 9, 4, 46, 10 ** 10.8, 10 ** 11.3, 4
A_PX = 0.02 ** 2  # pMpc^2
DISK = (np.hypot(*np.mgrid[-2:3, -2:3]) <= 2).astype(np.float64)
MB = np.logspace(10.8, 13.0, 23); RB = np.arange(RMIN, RMAX + 1, 3); RBk = (RB[:-1] + RB[1:]) / 2 * 20
AREA = np.array([((T2.R >= a) & (T2.R < b)).sum() for a, b in zip(RB[:-1], RB[1:])]) * A_PX  # pMpc^2 per annulus
FIELD = (T2.R >= RMIN) & (T2.R < RMAX)


def find(L):
    """L (n, 96, 96) log10 Sigma -> list per map of (R_px, M_40)"""
    out = []
    for m in L:
        S = 10 ** m.astype(np.float64); res = S - median_filter(S, MED, mode="nearest")
        Map = convolve(res, DISK, mode="nearest") * A_PX
        pk = (maximum_filter(res, 5) == res) & FIELD & (Map > MMIN)
        out.append((T2.R[pk], Map[pk]))
    return out


def stats(P):
    """cumulative N(>M) all and inner (R < 15 px = 300 kpc), radial counts n(R) for M > M_N; per map (n, ...)"""
    Nall = np.array([(M[:, None] > MB[None]).sum(0) for R, M in P]); Nin = np.array([(M[R < 15][:, None] > MB[None]).sum(0) for R, M in P])
    nR = np.array([np.histogram(R[M > M_N], RB)[0] / AREA for R, M in P])
    return Nall, Nin, nR


names = sys.argv[1:] or ["flow_hyb", "flow_hyb_mc"]
S, _ = C.load(); te = S["test"]; sig = te["sigma"]
_, first = np.unique(te["row"], return_index=True)
ST = {"CDM": stats(find(te["c"])), "true": stats(find(te["s"]))}; print("CDM and true SIDM done", flush=True)
for n in names:
    X = np.load(f"{C.HERE}/runs/dm_{n}/draws.npz")["draws"][:NDRAW].astype(np.float32)
    parts = [stats(find(X[d])) for d in range(len(X))]; ST[n] = tuple(np.mean([p[i] for p in parts], 0) for i in range(3)); print(n, "done", flush=True)
COL = {"CDM": "0.55", "true": "k"}; COL.update(dict(zip(names, ["#2a9d8f", "#d1495b", "#e9a03b"])))
mse = lambda a: (a.mean(0), a.std(0) / np.sqrt(len(a)))
fig, ax = plt.subplots(3, 5, figsize=(29, 15.5), constrained_layout=True); summ = {}
for i, s in enumerate(C.SIG):
    m = sig == s; A = ax[i]; summ[str(s)] = {}
    base = {q: ST["CDM"][q][m].mean(0) for q in range(3)}
    for nm, (Nall, Nin, nR) in ST.items():
        cl = COL[nm]; lw = 2.6 if nm in ("true", "CDM") else 1.8; ls = "-" if nm in ("true", "CDM") else "--"
        lab = nm if nm != "true" else f"true SIDM{s:g}"
        for j, (Q, qi) in enumerate(((Nall, 0), (Nin, 1))):
            mu, se = mse(Q[m]); A[j].plot(MB, mu, color=cl, lw=lw, ls=ls, label=lab); A[j].fill_between(MB, mu - se, mu + se, color=cl, alpha=0.15, lw=0)
        if nm != "CDM":
            A[2].plot(MB, Nall[m].mean(0) / base[0], color=cl, lw=lw, ls=ls, label=f"{lab} all R")
            A[2].plot(MB, Nin[m].mean(0) / base[1], color=cl, lw=lw, ls=":", label=f"{lab} R<300 kpc")
            A[4].plot(RBk, nR[m].mean(0) / base[2], color=cl, lw=lw, ls=ls, label=lab)
        mu, se = mse(nR[m]); A[3].plot(RBk, mu, color=cl, lw=lw, ls=ls, label=lab); A[3].fill_between(RBk, mu - se, mu + se, color=cl, alpha=0.15, lw=0)
        j11 = int(np.argmin(np.abs(MB - 10 ** 11.3))); j12 = int(np.argmin(np.abs(MB - 1e12)))
        summ[str(s)][nm] = dict(N_gt_11p3=float(Nall[m, j11].mean()), N_gt_12=float(Nall[m, j12].mean()), Nin_gt_11p3=float(Nin[m, j11].mean()),
                                n_inner=float(nR[m, :2].mean()), n_outer=float(nR[m, -3:].mean()))
    for j in (0, 1): A[j].set_xscale("log"); A[j].set_yscale("log"); A[j].set_xlabel("M_40 [Msun]"); A[j].set_ylim(0.02, None)
    A[0].set_ylabel(f"SIDM{s:g} ({m.sum()} test maps)\nN(> M_40) per map", fontsize=12); A[1].set_ylabel("N(> M_40) per map, R < 300 kpc")
    A[2].set_xscale("log"); A[2].set_xlabel("M_40 [Msun]"); A[2].set_ylabel("N(>M) / N_CDM(>M)"); A[2].axhline(1, color="0.6", lw=0.6); A[2].set_ylim(0, 1.6)
    A[3].set_yscale("log"); A[3].set_xlabel("R [kpc]"); A[3].set_ylabel(f"n(R) [pMpc⁻²], M_40 > 10^{np.log10(M_N):.1f}")
    A[4].set_xlabel("R [kpc]"); A[4].set_ylabel("n(R) / n_CDM(R)"); A[4].axhline(1, color="0.6", lw=0.6); A[4].set_ylim(0, 1.6)
    for a in A: a.legend(fontsize=8)
for j, t in enumerate(["subhalo mass function (all R)", "inner subhalo mass function (R < 300 kpc)", "relative to CDM", "radial number density", "radial density relative to CDM"]):
    ax[0, j].set_title(t, fontsize=13)
fig.suptitle(f"Projected subhaloes in the BAHAMAS DM maps (test split): peaks of Σ − median_{MED}px(Σ), aperture mass within 40 kpc, {RMIN * 20}–{RMAX * 20} kpc", fontsize=15)
fig.savefig(f"{C.HERE}/figures/subhalo_maps.png", dpi=62); json.dump(summ, open(f"{C.HERE}/runs/subhalo_maps_summary.json", "w"), indent=1)
for s, d in summ.items():
    for nm, r in d.items(): print(f"SIDM{s} {nm:12s} " + " ".join(f"{k}={v:.3f}" for k, v in r.items()))
print("wrote figures/subhalo_maps.png")
