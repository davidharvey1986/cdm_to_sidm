"""How much do stars and gas change from CDM to SIDM (BAHAMAS, matched pairs), and how much of it follows the dark matter?
Pairs: data/pairs_bar.npz + data/pairs_dm.npz, selection as dm_common (ok, |shift| <= 2 px, sigma > 0), all splits
(descriptive only, nothing is fitted). All maps log10 Sigma, smoothed 1.5 px; Delta = log SIDM - log CDM.
Figure 1 (figures/bar_change_stats.png), rows DM / stars / gas:
  stacked ratio <Sigma_SIDM>/<Sigma_CDM>(R); pixel rms of Delta(R) (pixels above 10x floor in both maps);
  per-pixel correlation of Delta with Delta_DM(R); per-pair mass ratio inside 50 and 500 kpc.
Figure 2 (figures/bar_change_maps.png): CDM, SIDM1 and Delta for DM, stars, gas for three clusters.
usage: python bar_change_check.py
"""
import os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
B = dict(np.load(f"{HERE}/data/pairs_bar.npz", allow_pickle=True)); Dd = {k: v for k, v in np.load(f"{HERE}/data/pairs_dm.npz", allow_pickle=True).items() if k in ("x", "y")}
sig, sh, xi = B["sigma"], B["shift"], B["xi"]
sel = B["ok_bar"] & (np.abs(sh).max(1) <= 2) & (sig > 0)
FL = {"DM": 11.0, "stars": 11.0, "gas": 11.0}  # stars: soft floor log10(Sigma + 1e11)
Y, X = np.indices((100, 100)); R = np.hypot(X - 49.5, Y - 49.5); RI = np.minimum(R.astype(int), 49); nR = np.bincount(RI.ravel())
rk = (np.arange(50) + 0.5) * 20; SIG = (0.1, 0.3, 1.0); COL = {0.1: "#3a6fb0", 0.3: "#2a9d8f", 1.0: "#d1495b"}
ring = lambda m: np.bincount(RI.ravel(), m.ravel(), 50)
A50, A500 = R < 2.5, R < 25


def get(ch, p):
    if ch == "DM": return Dd["x"][xi[p]], Dd["y"][p]
    k = {"stars": "s", "gas": "g"}[ch]; return B["x" + k][xi[p]], B["y" + k][p]


fig, ax = plt.subplots(3, 4, figsize=(24, 15), constrained_layout=True); summ = {}
for r, ch in enumerate(("DM", "stars", "gas")):
    for s in SIG:
        P = np.where(sel & (sig == s))[0]
        sx, sy, d1, d2, nn, cxy, cxx, cyy = (np.zeros(50) for _ in range(8)); m50, m500 = [], []
        for p in P:
            x, y = get(ch, p); xd, yd = (get("DM", p) if ch != "DM" else (x, y)); d = y - x; dd = yd - xd
            lx, ly = 10 ** x.astype(np.float64), 10 ** y.astype(np.float64); sx += ring(lx); sy += ring(ly)
            g = (x > FL[ch] + 1) & (y > FL[ch] + 1)
            d1 += ring(d * g); d2 += ring(d ** 2 * g); nn += ring(g.astype(float))
            cxy += ring(d * dd * g); cxx += ring(d ** 2 * g); cyy += ring(dd ** 2 * g)
            m50.append(np.log10(ly[A50].sum() / lx[A50].sum())); m500.append(np.log10(ly[A500].sum() / lx[A500].sum()))
        ok = nn > 200; mu = d1 / np.maximum(nn, 1)
        rms = np.where(ok, np.sqrt(np.maximum(d2 / np.maximum(nn, 1) - mu ** 2, 0)), np.nan)
        # correlation about zero-mean-per-ring pooled pixels
        md = cxy / np.maximum(nn, 1); corr = np.where(ok, md / np.sqrt(cxx / np.maximum(nn, 1) * cyy / np.maximum(nn, 1)), np.nan)
        ax[r, 0].plot(rk, sy / sx, color=COL[s], lw=2, label=f"SIDM{s:g}")
        ax[r, 1].plot(rk, rms, color=COL[s], lw=2, label=f"SIDM{s:g}")
        if ch != "DM": ax[r, 2].plot(rk, corr, color=COL[s], lw=2, label=f"SIDM{s:g}")
        bins = np.linspace(-0.4, 0.4, 81)
        ax[r, 3].hist(m50, bins, histtype="step", color=COL[s], lw=2, label=f"SIDM{s:g} <50 kpc")
        ax[r, 3].hist(m500, bins, histtype="step", color=COL[s], lw=1.2, ls="--", label=f"SIDM{s:g} <500 kpc")
        summ[(ch, s)] = (len(P), (sy / sx)[0], (sy / sx)[1], np.median(m50), np.std(m50), np.median(m500), np.std(m500), rms[1], rms[10], np.nanmean(corr[1:5]))
    ax[r, 0].axhline(1, color="0.6", lw=0.6, ls=":"); ax[r, 0].set_ylabel(f"{ch}\n<Σ_SIDM> / <Σ_CDM> (stacked)", fontsize=13)
    ax[r, 1].set_ylabel("rms of Δ per pixel [dex]"); ax[r, 2].set_ylabel("corr(Δ, Δ_DM) per pixel"); ax[r, 3].set_xlabel("log10 M_SIDM / M_CDM in aperture")
    for c in range(3): ax[r, c].set_xscale("log"); ax[r, c].set_xlabel("R [kpc]")
    for c in range(4): ax[r, c].legend(fontsize=8)
ax[0, 2].text(0.5, 0.5, "(DM: reference row)", ha="center", va="center", transform=ax[0, 2].transAxes, fontsize=13, color="0.5")
ax[1, 2].axhline(0, color="0.6", lw=0.6); ax[2, 2].axhline(0, color="0.6", lw=0.6)
for c, t in enumerate(["stacked profile ratio", "pixel scatter of Δ (both maps > 10× floor)", "does Δ follow Δ_DM?", "per-pair mass change (solid <50 kpc, dashed <500 kpc)"]):
    ax[0, c].set_title(t, fontsize=12)
fig.suptitle("BAHAMAS CDM → SIDM, matched pairs: change in dark matter, stars and gas (smoothed 1.5 px, Δ = log SIDM − log CDM)", fontsize=15)
fig.savefig(f"{HERE}/figures/bar_change_stats.png", dpi=60)
for (ch, s), v in summ.items():
    print(f"{ch:5s} SIDM{s:g} ({v[0]} pairs): stacked ratio R0 {v[1]:.3f} R1 {v[2]:.3f}; log M ratio <50kpc med {v[3]:+.3f} sd {v[4]:.3f}, "
          f"<500kpc med {v[5]:+.3f} sd {v[6]:.3f}; pixel rms R=30kpc {v[7]:.3f} R=210kpc {v[8]:.3f}; corr with dDM (30-90kpc) {v[9]:.2f}")

# example maps: three SIDM1 pairs spanning mass
P1 = np.where(sel & (sig == 1.0))[0]; lm = B["log10_M200"][xi[P1]]
ex = [P1[np.argsort(lm)[int(q * (len(P1) - 1))]] for q in (0.5, 0.9, 1.0)]
fig, ax = plt.subplots(len(ex), 9, figsize=(36, 4.3 * len(ex)), constrained_layout=True)
for r, p in enumerate(ex):
    for j, ch in enumerate(("DM", "stars", "gas")):
        x, y = get(ch, p); lo = {"DM": 12.5, "stars": 11.0, "gas": 12.0}[ch]; hi = {"DM": 15.8, "stars": 15.5, "gas": 14.8}[ch]
        ax[r, 3 * j].imshow(x, origin="lower", cmap="magma", vmin=lo, vmax=hi); ax[r, 3 * j + 1].imshow(y, origin="lower", cmap="magma", vmin=lo, vmax=hi)
        v = 0.3 if ch == "DM" else 0.6; ax[r, 3 * j + 2].imshow(y - x, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
        for c, t in enumerate(("CDM", "SIDM1", "Δ")):
            if r == 0: ax[r, 3 * j + c].set_title(f"{ch} {t}" + (f" (±{v})" if t == "Δ" else ""), fontsize=13)
    ax[r, 0].set_ylabel(f"z {B['redshift'][xi[p]]:.3f}, logM {B['log10_M200'][xi[p]]:.2f}", fontsize=12)
    for a in ax[r]: a.set_xticks([]); a.set_yticks([])
fig.suptitle("CDM vs SIDM1 for three clusters (median, 90th percentile and most massive): dark matter, stars, gas", fontsize=15)
fig.savefig(f"{HERE}/figures/bar_change_maps.png", dpi=50); print("wrote figures/bar_change_stats.png, bar_change_maps.png")
