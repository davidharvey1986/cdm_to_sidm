"""Enclosed-mass ratio log10 M_SIDM(<R) / M_CDM(<R) vs aperture radius for one cluster (default merger row 1951):
one true SIDM run, the noise-reduced response, the mass+z converters (BAHAMAS- and darkskies-trained, separate),
plus the BAHAMAS population (all pairs at that sigma: median and 16-84%).
usage: python3 plot_merger_mass.py [row]"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
row = int(sys.argv[1]) if len(sys.argv) > 1 else 1951
sys.argv = ["x", str(row)]
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(f"{HERE}/plot_merger_delta.py").read().split("cols = [")[0])
yy, xx = np.indices((96, 96)); rr = np.hypot(yy - 47.5, xx - 47.5); RA = np.arange(1, 49)
encl = lambda L: np.array([L[..., rr < a].sum(-1) for a in RA]).T  # (..., len(RA))
Lc = 10.0 ** c; Mc = encl(Lc)
ratio = lambda d: np.log10(encl(10.0 ** (c + np.asarray(d, float))) / Mc)
ok = (np.abs(D["shift"]).max(1) <= 2)
fig, ax = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True, sharey=True)
for i, s in enumerate(SIG):
    g = ax[i]; k = np.where(ok & (sig == s))[0]
    X = 10.0 ** x[xi[k], 2:98, 2:98].astype(float); Y = 10.0 ** y[k, 2:98, 2:98].astype(float)
    P = np.log10(np.stack([Y[:, rr < a].sum(1) / X[:, rr < a].sum(1) for a in RA], 1))
    g.fill_between(RA, *np.percentile(P, [16, 84], 0), color="0.85", lw=0, label=f"BAHAMAS population 16–84% (n={len(k)})")
    g.plot(RA, np.median(P, 0), color="0.5", lw=1.5, label="BAHAMAS population median")
    g.plot(RA, ratio(true[s]), "k", lw=2.2, label="this cluster: true SIDM run")
    if nr: g.plot(RA, ratio(nr[s]), color="#8e44ad", lw=1.8, label="this cluster: noise-reduced response")
    g.plot(RA, ratio(conv["bahamas", s]), "--", color="#3a6fb0", lw=2, label="converter, BAHAMAS-trained")
    g.plot(RA, ratio(conv["darkskies", s]), ":", color="#c0392b", lw=2.2, label="converter, darkskies-trained")
    g.axhline(0, color="0.4", lw=0.7); g.set_xscale("log"); g.set_xlabel("aperture radius R [px = 20 kpc]")
    g.set_title(f"σ = {s}"); g.set_ylim(-0.2, 0.08)
ax[0].set_ylabel("log10 M_SIDM(<R) / M_CDM(<R)"); ax[0].legend(fontsize=7, loc="lower right")
fig.suptitle(f"enclosed-mass ratio, BAHAMAS row {row} (merger, log M200 {m:.2f}, z {z})")
out = f"{HERE}/figures/merger_mass_{row}.png"; fig.savefig(out, dpi=100); print(out)
