"""All velocity-map channels for one halo/model/snap, three projections (rows).
usage: python3 plot_velmaps.py [dir] [halo]   (default: test_output, halo_00000001) -> <dir>/velmaps_<halo>.png"""
import os, sys, pickle
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/test_output"; H = sys.argv[2] if len(sys.argv) > 2 else "halo_00000001"
ext = [-1, 1, -1, 1]; kw = dict(origin="lower", interpolation="nearest", extent=ext)
fig, ax = plt.subplots(3, 5, figsize=(24, 14), constrained_layout=True)
for i, a_ in enumerate("xyz"):
    m, meta = pickle.load(open(f"{D}/{H}_{a_}.pkl", "rb"))
    vm = m[1] - np.nanmedian(m[1][40:60, 40:60])  # relative to the cluster's own bulk motion
    panels = [(np.log10(m[0]), "magma", None, None, "log10 Σ_DM, full depth [M☉/pMpc²]"),
              (vm, "RdBu_r", -700, 700, "mean LOS velocity − cluster bulk [km/s]"),
              (m[2], "viridis", 250, 850, "σ_los, full depth [km/s]"),
              (m[4], "viridis", 250, 850, "σ_los, ±R200 slab [km/s]"),
              (np.log10(m[0]) + np.log10(m[2] / 1e3), "inferno", None, None, "log10 Σ_DM × σ_los/1000 (rate proxy)")]
    for j, (img, cm, lo, hi, t) in enumerate(panels):
        im = ax[i, j].imshow(img, cmap=cm, vmin=lo, vmax=hi, **kw); fig.colorbar(im, ax=ax[i, j], shrink=0.75)
        if i == 0: ax[i, j].set_title(t, fontsize=11)
        ax[i, j].set_xlabel("pMpc")
    ax[i, 0].set_ylabel(f"projection {a_}\npMpc", fontsize=12)
fig.suptitle(f"dark-matter velocity maps, {H}, CDM z={meta['redshift']} (R200 = {meta['props']['R_200crit']:.2f} pMpc)", fontsize=14)
out = f"{D}/velmaps_{H}.png"; fig.savefig(out, dpi=80); print(out)
