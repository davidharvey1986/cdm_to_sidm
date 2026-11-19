"""Where are the densest pixels (log Sigma above THR) of one cluster (default merger row 1951) in CDM and each
SIDM model? Marks them on the CDM map, one colour per model, plus zooms on the regions holding the most dense pixels.
usage: python3 plot_merger_toppix.py [row] [thr]"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
HERE = os.path.dirname(os.path.abspath(__file__))
row = int(sys.argv[1]) if len(sys.argv) > 1 else 1951; THR = float(sys.argv[2]) if len(sys.argv) > 2 else 14.9
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sig = (D[k] for k in ("x", "y", "xi", "sigma"))
maps = {"CDM": x[row, 2:98, 2:98]}
for s in (0.1, 0.3, 1.0): maps[f"SIDM {s}"] = y[np.where((xi == row) & (sig == s))[0][0], 2:98, 2:98]
col = {"CDM": "w", "SIDM 0.1": "#2a9d8f", "SIDM 0.3": "#e9a03b", "SIDM 1.0": "#e74c3c"}
mk = {"CDM": "s", "SIDM 0.1": "o", "SIDM 0.3": "^", "SIDM 1.0": "D"}
off = {"CDM": (0, 0), "SIDM 0.1": (-0.22, -0.22), "SIDM 0.3": (0.22, -0.22), "SIDM 1.0": (0, 0.25)}
c = maps["CDM"]; lo, hi = np.percentile(c, [1, 99.9])
# dense regions = connected blobs of pixels above THR in any model (dilated)
anyhi = np.any([m > THR for m in maps.values()], 0)
from scipy.ndimage import binary_dilation
lab, nreg = label(binary_dilation(anyhi, iterations=2))
npx = [sum(int(((m > THR) & (lab == L)).sum()) for m in maps.values()) for L in range(1, nreg + 1)]
order = np.argsort(npx)[::-1] + 1  # regions with the most dense pixels (summed over models) first
cents = center_of_mass(anyhi, lab, order)
fig = plt.figure(figsize=(6 + 4.2 * min(nreg, 4), 6.2), constrained_layout=True)
gs = fig.add_gridspec(1, 1 + min(nreg, 4), width_ratios=[1.45] + [1] * min(nreg, 4))
a0 = fig.add_subplot(gs[0]); a0.imshow(c, cmap="magma", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
for k, m in maps.items():
    iy, ix = np.nonzero(m > THR)
    a0.scatter(ix + off[k][0], iy + off[k][1], marker=mk[k], s=22, facecolors="none", edgecolors=col[k], lw=1.3,
               label=f"{k}: {len(iy)} px > {THR}")
for j, (cy, cx) in enumerate(cents[:4]):
    a0.add_patch(plt.Rectangle((cx - 6.5, cy - 6.5), 13, 13, fill=False, ec="c", lw=1)); a0.text(cx + 7, cy + 7, str(j + 1), color="c", fontsize=12)
a0.legend(fontsize=8, loc="lower right", facecolor="0.2", labelcolor="w"); a0.set_title(f"CDM map, pixels with log Σ > {THR}")
a0.set_xticks([]); a0.set_yticks([])
for j, (cy, cx) in enumerate(cents[:4]):
    a = fig.add_subplot(gs[j + 1]); y0, x0 = int(round(cy)) - 6, int(round(cx)) - 6
    sub = c[max(y0, 0):y0 + 13, max(x0, 0):x0 + 13]
    a.imshow(sub, cmap="magma", vmin=lo, vmax=hi, origin="lower", interpolation="nearest",
             extent=[max(x0, 0) - 0.5, max(x0, 0) + sub.shape[1] - 0.5, max(y0, 0) - 0.5, max(y0, 0) + sub.shape[0] - 0.5])
    for k, m in maps.items():
        iy, ix = np.nonzero(m > THR); s_ = (abs(iy - cy) < 7) & (abs(ix - cx) < 7)
        a.scatter(ix[s_] + off[k][0], iy[s_] + off[k][1], marker=mk[k], s=90, facecolors="none", edgecolors=col[k], lw=1.8)
        for yy_, xx_ in zip(iy[s_], ix[s_]):
            pass
    r_ = np.hypot(cy - 47.5, cx - 47.5) * 20
    a.set_title(f"region {j + 1}: {r_:.0f} kpc from centre\nmax log Σ: " + ", ".join(f"{k.split()[-1]} {maps[k][max(y0,0):y0+13, max(x0,0):x0+13].max():.2f}" for k in maps), fontsize=8)
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"BAHAMAS row {row} (merger): location of the densest pixels in each model")
out = f"{HERE}/figures/merger_toppix_{row}.png"; fig.savefig(out, dpi=100); print(out, nreg, "regions")
