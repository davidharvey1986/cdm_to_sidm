"""Stellar surface-density profile, BAHAMAS test split: true SIDM and the full chain (stars-flow draw on a DM-flow
draw, runs/stars_hyb), one column per sigma. Linear Sigma_* = 10^l - 1e11, mean in 20 kpc rings per map, then the
median over maps; bands = 16-84% over maps (truth) and the spread of the per-draw medians (chain).
Figure figures/stellar_profile_chain.png. usage: python stellar_profile_chain.py [dm]   (dm: the DM profile, true SIDM vs runs/dm_flow_hyb, figures/dm_profile_chain.png)
"""
import sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import dm_common as C, two_scale as T2
RB = np.arange(0, 47); RK = (RB[:-1] + 0.5) * 20; LB = np.digitize(T2.R, RB) - 1; OK = (LB >= 0) & (LB < len(RK))
cnt = np.bincount(LB[OK], minlength=len(RK))
DM = len(sys.argv) > 1 and sys.argv[1] == "dm"
lin = (lambda l: 10 ** l.astype(np.float64)) if DM else (lambda l: np.maximum(10 ** l.astype(np.float64) - 1e11, 0))


def prof(L):
    """(n, 96, 96) log maps -> (n, nR) ring-mean Sigma_* [Msun/pMpc^2]"""
    return np.array([np.bincount(LB[OK], lin(l)[OK], minlength=len(RK)) / cnt for l in L])


S, _ = C.load(); te = S["test"]; sig = te["sigma"]; D = np.load(f"{C.HERE}/data/pairs_stars.npz", allow_pickle=True)
xs = D["x"][D["xi"][te["k"]]][:, 2:98, 2:98]; ys = D["y"][te["k"]][:, 2:98, 2:98]
X = np.load(f"{C.HERE}/runs/stars_hyb/draws.npz")["draws"].astype(np.float32)
if DM: xs, ys, X = te["c"], te["s"], np.load(f"{C.HERE}/runs/dm_flow_hyb/draws.npz")["draws"].astype(np.float32)
NM, SYM, OUT = ("dark matter", "Σ_DM", "dm_profile_chain") if DM else ("stellar", "Σ*", "stellar_profile_chain")
P = {"CDM": prof(xs), "true": prof(ys)}; PC = np.stack([prof(X[d]) for d in range(len(X))])
fig, ax = plt.subplots(2, 3, figsize=(18, 10), sharex=True, height_ratios=[2, 1], constrained_layout=True)
for i, s in enumerate(C.SIG):
    m = sig == s; A = ax[:, i]; base = np.median(P["true"][m], 0)
    A[1].axhspan(0.9, 1.1, color="0.88", lw=0, zorder=0, label="±10%"); A[1].axhspan(0.95, 1.05, color="0.7", lw=0, zorder=0, label="±5%")
    for nm, cl, lab in (("true", "k", f"true SIDM{s:g}"),):
        q = np.percentile(P[nm][m], [16, 50, 84], 0)
        A[0].plot(RK, q[1], color=cl, lw=2.6, label=lab); A[0].fill_between(RK, q[0], q[2], color=cl, alpha=0.12, lw=0)
        A[1].plot(RK, q[1] / base, color=cl, lw=2.6, label=lab)
    md = np.median(PC[:, m], 1); lo, mid, hi = np.percentile(md, [16, 50, 84], 0)
    A[0].plot(RK, mid, color="#d1495b", lw=2, ls="--", label=f"chain ({len(X)} draws)")
    A[1].plot(RK, mid / base, color="#d1495b", lw=2, ls="--", label="chain"); A[1].fill_between(RK, lo / base, hi / base, color="#d1495b", alpha=0.2, lw=0)
    A[0].set_xscale("log"); A[0].set_yscale("log"); A[0].set_title(f"SIDM{s:g} ({m.sum()} test maps)", fontsize=13)
    A[1].axhline(1, color="0.6", lw=0.6); A[1].set_ylim(0.85, 1.15); A[1].set_xlabel("R [kpc]")
    A[0].legend(fontsize=10); A[1].legend(fontsize=9)
    print(f"SIDM{s:g} Sigma/Sigma_trueSIDM at R=10,30,50,100,200 kpc: CDM " + " ".join(f"{np.median(P['CDM'][m], 0)[j] / base[j]:.2f}" for j in (0, 1, 2, 4, 9))
          + " | chain " + " ".join(f"{mid[j] / base[j]:.2f}" for j in (0, 1, 2, 4, 9)))
ax[0, 0].set_ylabel(f"{SYM} [Msun pMpc⁻²], median over maps"); ax[1, 0].set_ylabel("relative to true SIDM")
fig.suptitle(f"{NM.capitalize()} surface-density profile, BAHAMAS test split: true SIDM and the full chain" + ("" if DM else " (stars flow on DM-flow draws)"), fontsize=14)
fig.savefig(f"{C.HERE}/figures/{OUT}.png", dpi=70); print(f"wrote figures/{OUT}.png")
