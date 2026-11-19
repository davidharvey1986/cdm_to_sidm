"""One cluster (default merger row 1951): the three log-delta maps (SIDM 0.1, 0.3, 1 minus CDM) and their average.
usage: python3 plot_merger_avgdelta.py [row]"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); row = int(sys.argv[1]) if len(sys.argv) > 1 else 1951
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sig = (D[k] for k in ("x", "y", "xi", "sigma"))
SIG = (0.1, 0.3, 1.0); c = x[row, 2:98, 2:98]
d = {s: y[np.where((xi == row) & (sig == s))[0][0], 2:98, 2:98] - c for s in SIG}
avg = np.mean([d[s] for s in SIG], 0)
fig, ax = plt.subplots(1, 5, figsize=(22, 4.6), constrained_layout=True)
lo, hi = np.percentile(c, [1, 99.9]); kw = dict(origin="lower", interpolation="nearest")
ax[0].imshow(c, cmap="magma", vmin=lo, vmax=hi, **kw); ax[0].set_title("CDM")
for j, s in enumerate(SIG, 1):
    ax[j].imshow(d[s], cmap="RdBu_r", vmin=-.25, vmax=.25, **kw); ax[j].set_title(f"Δ, σ = {s}")
im = ax[4].imshow(avg, cmap="RdBu_r", vmin=-.25, vmax=.25, **kw); ax[4].set_title("average of the three Δ maps")
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.colorbar(im, ax=ax[1:], shrink=0.85, label="Δ = log10 SIDM − log10 CDM [dex]")
fig.suptitle(f"BAHAMAS row {row} (merger)")
out = f"{HERE}/figures/merger_avgdelta_{row}.png"; fig.savefig(out, dpi=95); print(out)
