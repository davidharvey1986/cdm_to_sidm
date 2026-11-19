"""One cluster (default merger row 1951): per-pixel delta log (SIDM - CDM) against the pixel's CDM density,
for SIDM 0.1, 0.3, 1; points coloured by distance from the centre, binned median and 16-84% band; last panel
overlays the three medians.
usage: python3 plot_merger_pixel.py [row]"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); row = int(sys.argv[1]) if len(sys.argv) > 1 else 1951
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sig = (D[k] for k in ("x", "y", "xi", "sigma"))
SIG = (0.1, 0.3, 1.0); c = x[row, 2:98, 2:98].ravel()
yy, xx = np.indices((96, 96)); r = np.hypot(yy - 47.5, xx - 47.5).ravel() * 0.02
d = {s: (y[np.where((xi == row) & (sig == s))[0][0], 2:98, 2:98]).ravel() - c for s in SIG}
bins = np.linspace(np.percentile(c, 0.5), c.max() + 1e-6, 16); mid = 0.5 * (bins[1:] + bins[:-1])
col = dict(zip(SIG, ("#2a9d8f", "#e9a03b", "#c0392b")))
fig, ax = plt.subplots(1, 4, figsize=(21, 5), constrained_layout=True, sharey=True)
for i, s in enumerate(SIG):
    g = ax[i]; sc = g.scatter(c, d[s], c=r, s=3, cmap="viridis", alpha=0.5, vmin=0, vmax=1.0, rasterized=True)
    idx = np.digitize(c, bins) - 1
    med = np.array([np.median(d[s][idx == b]) if (idx == b).sum() > 5 else np.nan for b in range(len(mid))])
    lo = np.array([np.percentile(d[s][idx == b], 16) if (idx == b).sum() > 5 else np.nan for b in range(len(mid))])
    hi = np.array([np.percentile(d[s][idx == b], 84) if (idx == b).sum() > 5 else np.nan for b in range(len(mid))])
    g.fill_between(mid, lo, hi, color="k", alpha=0.15, lw=0); g.plot(mid, med, "k", lw=2)
    ax[3].plot(mid, med, color=col[s], lw=2.2, label=f"σ = {s}"); ax[3].fill_between(mid, lo, hi, color=col[s], alpha=0.12, lw=0)
    g.axhline(0, color="0.5", lw=0.7); g.set_title(f"σ = {s}"); g.set_xlabel("CDM pixel log10 Σ [M☉/pMpc²]")
fig.colorbar(sc, ax=ax[:3], shrink=0.8, label="distance from centre [Mpc]")
ax[0].set_ylabel("Δ log = log10 Σ_SIDM − log10 Σ_CDM (per pixel)"); ax[0].set_ylim(-0.6, 0.6)
ax[3].axhline(0, color="0.5", lw=0.7); ax[3].legend(); ax[3].set_title("binned medians (bands 16–84%)"); ax[3].set_xlabel("CDM pixel log10 Σ")
fig.suptitle(f"BAHAMAS row {row} (merger): per-pixel Δ log vs pixel density")
out = f"{HERE}/figures/merger_pixel_{row}.png"; fig.savefig(out, dpi=95); print(out)
