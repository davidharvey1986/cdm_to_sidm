"""Velocity maps v1 (cluster's own FOF particles) vs v2 (all particles, deep5 / slab) for the BAHAMAS test clusters,
line of sight z (= stamp projection_0). Moments smoothed 1 px before sigma. usage: python3 compare_v1_v2.py"""
import os, numpy as np, matplotlib
from scipy.ndimage import gaussian_filter as G
matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__))
v1 = np.load(f"{H}/test_output/los2.npz"); v2 = np.load(f"{H}/test_output_v2/los2.npz")
def sig(M, j):
    W0, W1, W2 = (G(M[j + k].astype(float), 1) for k in range(3)); vm = W1 / W0; return np.sqrt(np.clip(W2 / W0 - vm * vm, 0, None))
fig, ax = plt.subplots(5, 4, figsize=(17, 20), constrained_layout=True); kw = dict(origin="lower", extent=[-1, 1, -1, 1], cmap="viridis", vmin=400, vmax=1600)
for i in range(5):
    maps = [(sig(v1["moments"][i], 4), "v1: own FOF, ±R200"), (sig(v2["moments"][i], 4), "v2: all DM, ±R200 (slab)"), (sig(v2["moments"][i], 0), "v2: all DM, ±5 pMpc (deep5)")]
    for j, (m, t) in enumerate(maps):
        im = ax[i, j].imshow(m, **kw); ax[i, j].set_title(t if i == 0 else "")
    fig.colorbar(im, ax=ax[i, :3], shrink=0.8, label="σ_los [km/s]")
    r = np.hypot(*np.indices((100, 100)) - 49.5).astype(int); cnt = np.bincount(r.ravel())
    for m, (lab, ls) in zip(maps, (("v1 own FOF", ":"), ("v2 slab", "-"), ("v2 deep5", "--"))):
        ax[i, 3].plot(np.arange(len(cnt)) * 20, np.bincount(r.ravel(), np.nan_to_num(m[0]).ravel()) / cnt, ls, lw=2, label=lab)
    ax[i, 3].set_xlabel("R [kpc]"); ax[i, 3].set_ylabel("σ_los [km/s]"); ax[i, 3].legend(fontsize=8)
    ax[i, 0].set_ylabel(f"fof {v2['fof'][i]}, log M {v2['log10_M200'][i]:.2f}")
fig.suptitle("BAHAMAS CDM+baryons z=0, LOS z: velocity dispersion with the three particle selections")
fig.savefig(f"{H}/test_output_v2/compare_v1_v2.png", dpi=70); print("ok")
