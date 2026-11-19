"""SIDM proxy (Sigma_DM * sigma_los * sigma/m) for the merger (pairs row 1951) against the true change, and a population
test: does the mean per-pixel change collapse onto one curve vs the proxy (it should if the proxy sets the response)?
usage: python3 plot_sidm_proxy.py -> unet/figures/sidm_proxy_merger.png, sidm_proxy_collapse.png"""
import os, numpy as np, matplotlib
from scipy.ndimage import gaussian_filter as G
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); UNET = os.path.join(HERE, "..", "..", "unet")
D = np.load(f"{UNET}/data/pairs.npz", allow_pickle=True); X = np.load(f"{UNET}/data/sidm_proxy_bahamas.npz")
x, y, xi, sig, sh = (D[k] for k in ("x", "y", "xi", "sigma", "shift"))
SIG = (0.1, 0.3, 1.0); C = 1.989e33 / 3.0857e24 ** 2; row = 1951; kw = dict(origin="lower", extent=[-1, 1, -1, 1])
# ---- merger
fig, ax = plt.subplots(2, 5, figsize=(26, 10), constrained_layout=True)
S, sv, base = X["Sigma_dm_slab"][row] * C, X["sigma_los_slab"][row], X["base_slab"][row]
for a_, img, cm, t in ((ax[0, 0], np.log10(S), "magma", "log Σ_DM [g/cm²] (CDM)"), (ax[1, 0], sv, "viridis", "σ_los [km/s] (CDM)")):
    im = a_.imshow(img, cmap=cm, **kw); fig.colorbar(im, ax=a_, shrink=0.8); a_.set_title(t)
lo, hi = np.nanpercentile(np.log10(base * 0.1), 1), np.nanpercentile(np.log10(base * 1.0), 99.9)
for j, s in enumerate(SIG, 1):
    im = ax[0, j].imshow(np.log10(base * s), cmap="inferno", vmin=lo, vmax=hi, **kw); fig.colorbar(im, ax=ax[0, j], shrink=0.8)
    ax[0, j].set_title(f"log proxy = log Σ σ_los σ/m [km/s], σ={s}")
    k = np.where((xi == row) & (sig == s))[0][0]
    im = ax[1, j].imshow(G(y[k] - x[row], 1.5), cmap="RdBu_r", vmin=-0.2, vmax=0.2, **kw); fig.colorbar(im, ax=ax[1, j], shrink=0.8)
    ax[1, j].set_title(f"true Δ log Σ (smoothed 1.5 px), σ={s}")
g = ax[0, 4]
for s, cl in zip(SIG, ("#2a9d8f", "#e9a03b", "#c0392b")):
    k = np.where((xi == row) & (sig == s))[0][0]; d = G(y[k] - x[row], 1.5).ravel(); p = np.log10(base * s).ravel()
    g.scatter(p, d, s=1, alpha=0.2, color=cl, label=f"σ={s}")
g.set_xlabel("log proxy [km/s]"); g.set_ylabel("Δ log Σ"); g.legend(markerscale=10); g.set_title("this cluster: Δ vs proxy, per pixel")
ax[1, 4].axis("off")
fig.suptitle("BAHAMAS merger z0.250_291: SIDM scattering-rate proxy (CDM run) vs the true SIDM change", fontsize=15)
fig.savefig(f"{UNET}/figures/sidm_proxy_merger.png", dpi=70)
# ---- population collapse test (all pixels, |shift|<=2, 2:98 crop), with the SIDM-run noise kept
ok = (np.abs(sh).max(1) <= 2) & (sig > 0)
fig, ax = plt.subplots(1, 3, figsize=(20, 5.5), constrained_layout=True)
for s, cl in zip(SIG, ("#2a9d8f", "#e9a03b", "#c0392b")):
    k = np.where(ok & (sig == s))[0]; c = xi[k]; good = np.isfinite(X["base_slab"][c]).all((1, 2)); k, c = k[good], c[good]
    d = np.stack([G(y[i] - x[j], 1.5) for i, j in zip(k, c)])[:, 2:98, 2:98].ravel()
    Sg = (X["Sigma_dm_slab"][c] * C)[:, 2:98, 2:98].ravel(); pr = (X["base_slab"][c] * s)[:, 2:98, 2:98].ravel()
    for a_, v, lab in ((ax[0], np.log10(Sg), "log Σ_DM [g/cm²]"), (ax[1], np.log10(pr), "log proxy = Σ σ_los σ/m [km/s]"), (ax[2], np.log10(Sg * s), "log τ = Σ σ/m (no velocity)")):
        b = np.linspace(np.nanpercentile(v, 2), np.nanpercentile(v, 99.99), 30); m_ = 0.5 * (b[1:] + b[:-1]); idx = np.digitize(v, b) - 1
        mu = [d[idx == q].mean() if (idx == q).sum() > 500 else np.nan for q in range(len(m_))]
        a_.plot(m_, mu, "o-", ms=3, color=cl, label=f"σ = {s}"); a_.set_xlabel(lab)
for a_, t in zip(ax, ("vs density alone (no σ, no velocity)", "vs proxy Σ σ_los σ/m", "vs optical depth Σ σ/m")):
    a_.axhline(0, color="0.5", lw=0.7); a_.set_ylabel("mean Δ log Σ (SIDM − CDM, smoothed 1.5 px)"); a_.legend(); a_.set_title(t)
fig.suptitle("BAHAMAS, all clusters: does the SIDM change collapse onto one curve of the proxy?  (collapse = curves for the three σ overlap)")
fig.savefig(f"{UNET}/figures/sidm_proxy_collapse.png", dpi=90); print("ok")
