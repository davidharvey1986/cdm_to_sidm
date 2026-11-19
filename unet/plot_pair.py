"""Show one raw (un-normalised) training pair from data/pairs.npz.
usage: python3 plot_pair.py [sigma=1.0] [rank=0.25]   rank: mass quantile of the val cluster picked"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
q = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
D = np.load("data/pairs.npz", allow_pickle=True)
x, y, xi, sigma, shift, split = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "split"))
full = [n for n in np.unique(xi[split[xi] == "val"]) if set(sigma[xi == n]) >= {0.1, 0.3, 1.0}]
n = sorted(full, key=lambda n: -x[n][45:55, 45:55].mean())[int(q * len(full))]
p = np.where((xi == n) & (sigma == S))[0][0]
lc, ls = x[n], y[p]; dl = ls - lc
print(f"cluster {D['match_id'][n]} proj {D['projection'][n]} z={D['redshift'][n]} log10M200={D['log10_M200'][n]:.2f}; shift {shift[p]}")
for nm, v in (("CDM", 10**lc), ("SIDM", 10**ls)):
    print(f"  {nm:5s} linear: min {v.min():.3e} median {np.median(v):.3e} max {v.max():.3e} sum {v.sum():.3e}")
print(f"  log10 CDM range {lc.min():.2f}..{lc.max():.2f}; delta range {dl.min():.2f}..{dl.max():.2f}, centre px {dl[49:51,49:51].mean():+.3f}")
fig, ax = plt.subplots(1, 5, figsize=(19, 4), facecolor="white")
vmax = (10**lc).max()
for a, v, t in ((ax[0], 10**lc, "CDM, linear"), (ax[1], 10**ls, f"SIDM{S:g}, linear")):
    im = a.imshow(v, origin="lower", cmap="magma", vmin=0, vmax=vmax); fig.colorbar(im, ax=a, fraction=0.046); a.set_title(t)
lo, hi = np.percentile(lc, 0.5), lc.max()
for a, v, t in ((ax[2], lc, "log10 CDM"), (ax[3], ls, f"log10 SIDM{S:g} (re-centred)")):
    im = a.imshow(v, origin="lower", cmap="magma", vmin=lo, vmax=hi); fig.colorbar(im, ax=a, fraction=0.046); a.set_title(t)
im = ax[4].imshow(dl, origin="lower", cmap="RdBu_r", vmin=-0.4, vmax=0.4); fig.colorbar(im, ax=ax[4], fraction=0.046)
ax[4].set_title("delta = log10 SIDM - log10 CDM (target)")
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"{D['match_id'][n]}  projection {D['projection'][n]}  z = {D['redshift'][n]}  log10 M200 = {D['log10_M200'][n]:.2f}", y=1.0)
fig.tight_layout(); fig.savefig("../figures/raw_pair.png", dpi=110, bbox_inches="tight")
