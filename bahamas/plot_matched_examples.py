"""10x4 grid of random samples from bahamas_matched.pkl: rows = clusters,
columns = CDM, SIDM0.1, SIDM0.3, SIDM1 (+baryons) total-mass maps, one shared
log colour scale per row; missing channels (mask False) shown as empty panels.
usage: python3 plot_matched_examples.py [seed=0] [n_models: 2, 3, 4 or all]
Output: ../figures/bahamas_matched_examples[_<n>of4].png"""
import os, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas"
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
nsel = sys.argv[2] if len(sys.argv) > 2 else "all"
meta, im = pickle.load(open(f"{IMG}/bahamas_matched.pkl", "rb"))
pool = np.arange(len(im)) if nsel == "all" else np.where(meta["n_models"] == int(nsel))[0]
rows = np.sort(np.random.default_rng(seed).choice(pool, 10, replace=False))
labels = ["CDM", "SIDM0.1", "SIDM0.3", "SIDM1"]
INK, MUTED, EMPTY = "#1f1f1e", "#6b6a63", "#f0efec"

fig, axes = plt.subplots(10, 4, figsize=(7.6, 19.4), facecolor="white",
                         gridspec_kw=dict(wspace=0.04, hspace=0.12))
for r, i in enumerate(rows):
    vmin, vmax = np.nanpercentile(im[i], [1, 99.9])
    for c in range(4):
        ax = axes[r, c]
        if meta["mask"][i, c]:
            ax.imshow(im[i, c], origin="lower", cmap="magma", norm=LogNorm(vmin, vmax))
        else:
            ax.set_facecolor(EMPTY)
            ax.text(0.5, 0.5, "no stamp", ha="center", va="center", color=MUTED,
                    fontsize=9, transform=ax.transAxes)
            ax.set_aspect("equal")
            for s in ax.spines.values():
                s.set_color("#c3c2b7")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(labels[c], fontsize=11, color=INK)
        if c == 0:
            ax.set_ylabel(f"z={meta['redshift'][i]:.3f}  p{meta['projection'][i]}\n"
                          f"log M200={meta['log10_M200'][i]:.2f}", fontsize=8, color=MUTED)
what = "10 random clusters" if nsel == "all" else f"10 random clusters with {nsel} of 4 models"
fig.suptitle(f"bahamas_matched.pkl: {what} (total mass, +baryons)",
             fontsize=11, color=INK, y=0.992)
fig.subplots_adjust(top=0.965, bottom=0.01, left=0.08, right=0.99)
out = f"{HERE}/../figures/bahamas_matched_examples{'' if nsel == 'all' else f'_{nsel}of4'}.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(os.path.normpath(out), "samples", rows.tolist())
