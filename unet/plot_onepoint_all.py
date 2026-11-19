"""One-point function (pixel PDF of log10 Sigma, central 96x96 px) averaged over all BAHAMAS clusters, for each
SIDM model against CDM of the same clusters; ratio to CDM with bootstrap-over-clusters bands; high-density tail
per map. Also the two-scale converter (BAHAMAS- and darkskies-trained, mass+z template + small-scale map) for sigma = 1.
usage: python3 plot_onepoint_all.py [ymin ymax of the ratio panel]  -> figures/onepoint_all[_y...].png"""
import os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import two_scale as T2
HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sig, sh, zz, M = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "redshift", "log10_M200"))
import sys; YL = tuple(map(float, sys.argv[1:3])) if len(sys.argv) > 2 else (0.3, 1.4)
SIG = (0.1, 0.3, 1.0); b = np.linspace(12.4, 15.8, 69); mid = 0.5 * (b[1:] + b[:-1]); rng = np.random.default_rng(0)
col = dict(zip(SIG, ("#2a9d8f", "#e9a03b", "#c0392b")))
H = lambda maps: np.stack([np.histogram(m, b)[0] for m in maps])  # per-map histograms
MT = json.load(open(f"{HERE}/runs/mass_template/summary.json"))


def convert(c, s, z, Mh, tag):
    J = json.load(open(f"{HERE}/runs/two_scale_{tag}_nocnn/template.json")); C = np.array(MT[tag]["mass+z"]["C"]); p = J["template"]["p"]
    out = np.empty_like(c)
    for i in range(0, len(c), 256):
        cc = c[i:i + 256]; bp = C[0][None] + C[1][None] * (Mh[i:i + 256, None] - 14.5) + C[2][None] * (z[i:i + 256, None] - 0.2)
        Lc = T2.smooth(cc); Sc = cc - Lc
        out[i:i + 256] = cc + (s ** p) * bp[:, T2.RB] + (T2.apply_small(Sc, np.full(len(cc), s), J["small"]) - Sc)
    return out


fig, ax = plt.subplots(1, 3, figsize=(19, 5.2), constrained_layout=True)
for s in SIG:
    k = np.where((np.abs(sh).max(1) <= 2) & (sig == s))[0]
    c = x[xi[k], 2:98, 2:98].astype(np.float64); t = y[k, 2:98, 2:98].astype(np.float64)
    hc, ht = H(c), H(t); n = len(k)
    if s == SIG[0]:
        ax[0].step(mid, hc.sum(0) / hc.sum() / (b[1] - b[0]), where="mid", color="k", lw=2.4, label="CDM")
    ax[0].step(mid, ht.sum(0) / ht.sum() / (b[1] - b[0]), where="mid", color=col[s], lw=1.6, label=f"SIDM {s}")
    rat = (ht.sum(0) + 0.5) / (hc.sum(0) + 0.5)
    bs = [];  
    for _ in range(200):
        j = rng.integers(0, n, n); bs.append((ht[j].sum(0) + 0.5) / (hc[j].sum(0) + 0.5))
    lo, hi = np.percentile(bs, [16, 84], 0); ok = hc.sum(0) > 30
    ax[1].fill_between(mid[ok], lo[ok], hi[ok], color=col[s], alpha=0.25, lw=0); ax[1].plot(mid[ok], rat[ok], color=col[s], lw=2, label=f"SIDM {s} / CDM")
    cum = lambda h: h[:, ::-1].cumsum(1)[:, ::-1].sum(0) / n
    if s == SIG[0]: ax[2].plot(b[:-1], cum(hc), "k", lw=2.4, label="CDM")
    ax[2].plot(b[:-1], cum(ht), color=col[s], lw=1.8, label=f"SIDM {s}")
    if s == 1.0:
        for tag, ls in (("bahamas", "--"), ("darkskies", ":")):
            hv = H(convert(c, s, zz[xi[k]], M[xi[k]], tag)); r2 = (hv.sum(0) + 0.5) / (hc.sum(0) + 0.5)
            ax[1].plot(mid[ok], r2[ok], color="k", ls=ls, lw=1.6, label=f"converter σ=1 ({tag}-trained)")
            ax[2].plot(b[:-1], cum(hv), color="k", ls=ls, lw=1.4, label=f"converter σ=1 ({tag})")
ax[0].set_yscale("log"); ax[0].set_xlabel("pixel log10 Σ [M☉/pMpc²]"); ax[0].set_ylabel("PDF"); ax[0].set_title("one-point function, all clusters"); ax[0].legend()
ax[1].axhline(1, color="0.5", lw=0.7); ax[1].set_xlabel("pixel log10 Σ"); ax[1].set_ylabel("PDF ratio to CDM (bands: bootstrap over clusters)")
ax[1].set_title("SIDM / CDM"); ax[1].set_ylim(*YL); ax[1].legend(fontsize=7)
ax[2].set_yscale("log"); ax[2].set_xlim(14.3, 15.7); ax[2].set_ylim(1e-3, 300); ax[2].set_xlabel("pixel log10 Σ"); ax[2].set_ylabel("pixels per map above")
ax[2].set_title("high-density tail N(> log Σ) per map"); ax[2].legend(fontsize=7)
fig.suptitle("BAHAMAS: one-point function of the 96×96 px maps, averaged over all clusters (~3,300 per σ)")
out = f"{HERE}/figures/onepoint_all" + ("" if len(sys.argv) < 3 else f"_y{YL[0]}-{YL[1]}") + ".png"; fig.savefig(out, dpi=100); print(out)
