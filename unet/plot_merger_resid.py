"""Merger (pairs row 1951): residual true SIDM - converted (two-scale, darkskies-trained, no CNN) vs the no-conversion
residual true SIDM - CDM, per pixel and smoothed 3 px, plus azimuthal profiles. -> figures/merger_resid_1951.png"""
import os, json
import numpy as np, matplotlib
from scipy.ndimage import gaussian_filter as G
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import two_scale as T2
HERE = os.path.dirname(os.path.abspath(__file__)); row = 1951
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True); x, y, xi, sig = D["x"], D["y"], D["xi"], D["sigma"]
z = float(D["redshift"][row]); c = x[row, 2:98, 2:98].astype(float)
J = json.load(open(f"{HERE}/runs/two_scale_darkskies_nocnn/template.json")); SIG = (0.1, 0.3, 1.0)
r = np.hypot(*np.indices((96, 96)) - 47.5); rb = r.astype(int); cnt = np.bincount(rb.ravel())
prof = lambda m: np.bincount(rb.ravel(), m.ravel()) / cnt
fig, ax = plt.subplots(3, 5, figsize=(24, 14), constrained_layout=True)
kw = dict(origin="lower", interpolation="nearest", cmap="RdBu_r", extent=[-0.96, 0.96, -0.96, 0.96])
for i, s in enumerate(SIG):
    t = y[np.where((xi == row) & (sig == s))[0][0], 2:98, 2:98].astype(float)
    Lc = T2.smooth(c[None]); Sc = c[None] - Lc
    cv = (c[None] + T2.predict_large(Lc, np.array([s]), np.array([z]), J["template"]) + T2.apply_small(Sc, np.array([s]), J["small"]) - Sc)[0]
    R0, R1 = t - c, t - cv; a = ax[i]
    for q, (m, lim, ttl) in enumerate([(R0, .3, "true − CDM  (no conversion)"), (R1, .3, "true − converted"),
                                       (G(R0, 3), .1, "true − CDM, smoothed 3 px"), (G(R1, 3), .1, "true − converted, smoothed 3 px")]):
        im = a[q].imshow(m, vmin=-lim, vmax=lim, **kw); a[q].set_xticks([]); a[q].set_yticks([])
        a[q].set_title(f"{ttl}\nrms {m.std():.3f}, mean {m.mean():+.3f} dex", fontsize=11)
        if q in (1, 3): fig.colorbar(im, ax=a[q], shrink=0.8, label="dex")
    a[0].set_ylabel(f"SIDM{s:g}", fontsize=15)
    R = np.arange(len(cnt)) * 20 + 10
    a[4].plot(R, prof(R0), color="0.5", lw=1.8, label="true − CDM"); a[4].plot(R, prof(R1), color="#c0392b", lw=2, label="true − converted")
    a[4].axhline(0, color="k", lw=0.6); a[4].set_xscale("log"); a[4].set_ylim(-0.1, 0.1); a[4].set_xlabel("R [kpc]"); a[4].set_ylabel("azimuthal mean [dex]")
    if i == 0: a[4].legend(fontsize=10); a[4].set_title("azimuthal mean residual", fontsize=11)
fig.suptitle(f"Merger {D['match_id'][row]} (z = {z}): residual after conversion (darkskies-trained converter)", fontsize=15)
out = f"{HERE}/figures/merger_resid_{row}.png"; fig.savefig(out, dpi=80); print(out)
