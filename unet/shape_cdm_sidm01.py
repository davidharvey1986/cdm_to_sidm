"""CDM vs SIDM projected dark-matter halo shapes, BAHAMAS vs darkskies, one sigma per suite (default 0.1 for both, the only
sigma both suites have).
Reads runs/shape_suites_cache.npz (shape_suites.py). Panels: mean eps(R) for all masses and for log10 M200 > 14.7,
and the paired change eps_SIDM - eps_CDM. Bands: 1 sigma error on the mean. Output figures/shape_cdm_sidm<bh>_<ds>.png
usage: python shape_cdm_sidm01.py [--bh 0.3] [--ds 0.2]
"""
import argparse
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from halo_shape import RK
ap = argparse.ArgumentParser(); ap.add_argument("--bh", type=float, default=0.1); ap.add_argument("--ds", type=float, default=0.1); A = ap.parse_args()
C = np.load("runs/shape_suites_cache.npz"); MCUT = 14.7; SIG = {"BAHAMAS": A.bh, "darkskies": A.ds}
SU = {"BAHAMAS": ("BAHAMAS DM", "#3a6fb0"), "darkskies": ("darkskies DM", "#e07b39")}
mse = lambda e: (np.nanmean(e, 0), np.nanstd(e, 0) / np.sqrt(np.isfinite(e).sum(0)))
fig, ax = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
for lab, (key, col) in SU.items():
    eC, lmc, eY, dE, sg, lm = (C[f"{key}__{q}"] for q in ("eC", "logM_c", "eY", "dE", "sig", "logM"))
    s1 = np.isclose(sg, SIG[lab]); nm1 = f"SIDM{SIG[lab]:g}"
    for j, (kc, ky, tag) in enumerate([(np.ones_like(lmc, bool), s1, "all"), (lmc > MCUT, s1 & (lm > MCUT), "hi")]):
        for e, ls, nm in ((eC[kc], "-", "CDM"), (eY[ky], "--", nm1)):
            m, se = mse(e); ax[j].plot(RK, m, color=col, ls=ls, lw=2.4, label=f"{lab} {nm} ({len(e)})"); ax[j].fill_between(RK, m - se, m + se, color=col, alpha=0.18, lw=0)
        m, se = mse(dE[ky]); ax[2].plot(RK, m, color=col, lw=2.4, ls="-" if tag == "all" else ":", marker="o", ms=4,
                                         label=f"{lab} {nm1}, {'all masses' if tag == 'all' else f'logM > {MCUT}'} ({ky.sum()})")
        ax[2].fill_between(RK, m - se, m + se, color=col, alpha=0.15, lw=0)
for a, t in zip(ax, ("all masses", f"log10 M200 > {MCUT}", "paired change ε_SIDM − ε_CDM")):
    a.set_xscale("log"); a.set_xlabel("R (semi-major) [kpc]"); a.set_title(t, fontsize=14); a.legend(fontsize=10)
ax[0].set_ylabel("mean ε = 1 − b/a within R"); ax[1].set_ylabel("mean ε = 1 − b/a within R"); ax[2].set_ylabel("Δε"); ax[2].axhline(0, color="0.5", lw=0.8)
for a in ax[:2]: a.set_ylim(0, 0.47)
fig.suptitle(f"Dark-matter halo shapes, CDM vs SIDM: BAHAMAS SIDM{A.bh:g} (blue) vs darkskies SIDM{A.ds:g} (orange); |los| < 5 pMpc, smoothed 30 kpc; R ≤ 60 kpc is pixel-limited", fontsize=14)
out = "figures/shape_cdm_sidm01.png" if A.bh == A.ds == 0.1 else f"figures/shape_cdm_sidm_bh{A.bh:g}_ds{A.ds:g}.png"
fig.savefig(out, dpi=70); print("wrote", out)
for lab, (key, _) in SU.items():
    k = np.isclose(C[f"{key}__sig"], SIG[lab]) & (C[f"{key}__logM"] > MCUT); m, se = mse(C[f"{key}__dE"][k])
    print(f"{lab} SIDM{SIG[lab]:g} logM>{MCUT} ({k.sum()}): " + " ".join(f"R{r}={a:+.3f}±{b:.3f}" for r, a, b in zip(RK, m, se) if r >= 100))
