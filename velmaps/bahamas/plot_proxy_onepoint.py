"""One-point function (pixel PDF) of the SIDM proxy log(Sigma_DM * sigma_los * sigma/m) over all BAHAMAS CDM stamps,
for sigma = 0.1, 0.3, 1 (a pure shift by log sigma), plus its two ingredients and the split by redshift.
usage: python3 plot_proxy_onepoint.py -> unet/figures/proxy_onepoint.png"""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); UNET = os.path.join(HERE, "..", "..", "unet")
X = np.load(f"{UNET}/data/sidm_proxy_bahamas.npz"); D = np.load(f"{UNET}/data/pairs.npz", allow_pickle=True)
C = 1.989e33 / 3.0857e24 ** 2; good = np.isfinite(X["base_slab"]).all((1, 2))
base = X["base_slab"][good][:, 2:98, 2:98]; S = X["Sigma_dm_slab"][good][:, 2:98, 2:98] * C; sv = X["sigma_los_slab"][good][:, 2:98, 2:98]
zz = D["redshift"][good]
fig, ax = plt.subplots(1, 4, figsize=(24, 5.2), constrained_layout=True)
b = np.linspace(-2.5, 2.8, 107); m = 0.5 * (b[1:] + b[:-1])
for s, cl in zip((0.1, 0.3, 1.0), ("#2a9d8f", "#e9a03b", "#c0392b")):
    h = np.histogram(np.log10(base * s), b, density=True)[0]; ax[0].step(m, h, where="mid", color=cl, lw=2, label=f"σ = {s}")
    tail = np.sort(np.log10(base * s).ravel())[::-1]; ax[1].plot(tail, np.arange(1, len(tail) + 1) / len(base), color=cl, lw=2, label=f"σ = {s}")
ax[0].set_yscale("log"); ax[0].set_xlabel("log10 proxy = Σ_DM σ_los σ/m [km/s]"); ax[0].set_ylabel("PDF"); ax[0].legend(); ax[0].set_title("one-point function of the proxy")
ax[1].set_yscale("log"); ax[1].set_xlim(0.5, 2.8); ax[1].set_ylim(1e-3, 3e3); ax[1].set_xlabel("log10 proxy [km/s]"); ax[1].set_ylabel("pixels per map above")
ax[1].set_title("high tail N(> proxy) per map"); ax[1].legend()
for v, lab, a_ in ((np.log10(S), "log10 Σ_DM [g/cm²]", ax[2]), (sv, "σ_los [km/s]", ax[2].twiny())):
    pass
h1 = np.histogram(np.log10(S), np.linspace(-4, 0.3, 87), density=True); ax[2].step(0.5 * (h1[1][1:] + h1[1][:-1]), h1[0], where="mid", color="k", lw=2)
ax[2].set_yscale("log"); ax[2].set_xlabel("log10 Σ_DM [g/cm²]"); ax[2].set_title("ingredient 1: dark-matter surface density")
for zv, cl in zip(sorted(set(zz)), ("#1b4f72", "#2e86c1", "#85c1e9", "#d6eaf8")[::-1]):
    h = np.histogram(sv[zz == zv], np.linspace(0, 1600, 81), density=True); ax[3].step(0.5 * (h[1][1:] + h[1][:-1]), h[0], where="mid", lw=2, color=cl, label=f"z = {zv}")
ax[3].set_xlabel("σ_los [km/s]"); ax[3].set_title("ingredient 2: velocity dispersion, by redshift"); ax[3].legend()
fig.suptitle(f"BAHAMAS SIDM proxy, one-point function over {good.sum()} CDM stamps (96×96 px each, ±R200, moments smoothed 1.5 px)")
fig.savefig(f"{UNET}/figures/proxy_onepoint.png", dpi=90)
print("median / 99th / 99.99th pct of log proxy at sigma=1:", np.round(np.percentile(np.log10(base), [50, 99, 99.99]), 2))
print("spread (std) of log Sigma_DM", np.log10(S).std().round(3), " of log sigma_los", np.log10(sv[sv > 0]).std().round(3))
