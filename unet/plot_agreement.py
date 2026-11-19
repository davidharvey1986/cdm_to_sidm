"""Show the local-agreement weights for one pair (same cluster as plot_pair.py).
usage: python3 plot_agreement.py [sigma=1.0] [rank=0.25]"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from make_agreement_weights import agreement_weight
S = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
q = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
D = np.load("data/pairs.npz", allow_pickle=True)
x, y, xi, sigma, split = (D[k] for k in ("x", "y", "xi", "sigma", "split"))
full = [n for n in np.unique(xi[split[xi] == "val"]) if set(sigma[xi == n]) >= {0.1, 0.3, 1.0}]
n = sorted(full, key=lambda n: -x[n][45:55, 45:55].mean())[int(q * len(full))]
p = np.where((xi == n) & (sigma == S))[0][0]
c, t = x[n].astype(float), y[p].astype(float)
w, rho, f = agreement_weight(c, t)
fig, ax = plt.subplots(2, 3, figsize=(14, 9), facecolor="white")
lo, hi = np.percentile(c, 0.5), c.max()
panels = [(c, "log10 CDM", "magma", lo, hi), (t, f"log10 SIDM{S:g}", "magma", lo, hi),
          (t - c, "delta (dex)", "RdBu_r", -0.4, 0.4),
          (rho, "local correlation of high-passed maps", "PuOr_r", -1, 1),
          (f, "structure fraction f (0 smooth, 1 subhalo)", "Greys", 0, 1),
          (w, "loss weight w", "viridis", 0, 2)]
for a, (m, tt, cm, v0, v1) in zip(ax.ravel(), panels):
    im = a.imshow(m, origin="lower", cmap=cm, vmin=v0, vmax=v1); a.set_title(tt, fontsize=10)
    fig.colorbar(im, ax=a, fraction=0.046); a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"{D['match_id'][n]}  projection {D['projection'][n]}  z = {D['redshift'][n]}  sigma = {S:g}   "
             f"(weight < 0.5: {(w < 0.5).mean():.0%} of pixels, > 1.5: {(w > 1.5).mean():.0%})")
fig.tight_layout(); fig.savefig("../figures/agreement_weights.png", dpi=105)
