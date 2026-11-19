"""Model proxy for the line-of-sight velocity dispersion in each pixel (no velocity maps exist yet).

Main halo = NFW with M200c from the catalogue, c200 from Duffy et al. (2008), isotropic Jeans equation ->
projected sigma_los(R)/V200 as a function of R/r200 and c (tabulated). Pixel R = distance from the stamp centre,
20 pkpc pixels (BAHAMAS assumed the same; its profiles match darkskies). One overall amplitude A is calibrated on the
darkskies VELOCIraptor aperture dispersions (Aperture_veldisp_{10,30,100}_kpc, CDM, all snaps/projections).
Then the test: at fixed CDM pixel density, does the SIDM change depend on sigma_v (and on the rate proxy
Sigma * sigma_v)?  BAHAMAS sigma = 1 and darkskies sigma = 0.2, held-out + all pixels with r > 0.
usage: python3 sigma_v_proxy.py   -> data/sigma_v_proxy_{bahamas,darkskies}.npz, figures/sigma_v_proxy*.png
"""
import os, pickle
import numpy as np, pandas as pd, matplotlib
from scipy.integrate import quad
matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
G = 4.30091e-6; h = 0.7; OM = 0.3; PIX = 20.0  # kpc km^2 s^-2 Msun^-1; pkpc per pixel
rho_c = lambda z: 2.775e2 * h ** 2 * (OM * (1 + z) ** 3 + 1 - OM)  # Msun / kpc^3
c_duffy = lambda M, z: 5.71 * (M * h / 2e12) ** -0.084 * (1 + z) ** -0.47

# ---- dimensionless NFW projected dispersion table: s(X; c) = sigma_los / V200, X = R / r200
CG = np.linspace(2.5, 9.0, 14); XG = np.logspace(-2.7, 0.6, 60)
def tab(c):
    m = lambda x: (np.log(1 + c * x) - c * x / (1 + c * x)) / (np.log(1 + c) - c / (1 + c))
    rho = lambda x: 1.0 / (c * x * (1 + c * x) ** 2)
    # sigma_r^2 / V200^2 = (1/rho) int_x^inf rho m / x'^2 dx'
    sr2 = lambda x: quad(lambda t: rho(t) * m(t) / t ** 2, x, 50, limit=200)[0] / rho(x)
    xs = np.logspace(-3.2, np.log10(50), 400); S2 = np.array([sr2(t) for t in xs])
    out = []
    for X in XG:
        u = np.logspace(np.log10(X * 1.0001), np.log10(50), 300)
        w = rho(u) * u / np.sqrt(u ** 2 - X ** 2)
        num = np.trapz(w * np.interp(u, xs, S2), u); den = np.trapz(w, u)
        out.append(np.sqrt(num / den))
    return np.array(out)
TAB = np.array([tab(c) for c in CG])  # (nc, nX)


def s_los(X, c):
    ci = np.clip(np.interp(c, CG, np.arange(len(CG))), 0, len(CG) - 1.001); i0 = int(ci); f = ci - i0
    return np.exp(np.interp(np.log(np.clip(X, XG[0], XG[-1])), np.log(XG), (1 - f) * np.log(TAB[i0]) + f * np.log(TAB[i0 + 1])))


def halo(M, z):
    r200 = (3 * M / (4 * np.pi * 200 * rho_c(z))) ** (1 / 3); V200 = np.sqrt(G * M / r200); return r200, V200, c_duffy(M, z)


N = 100; yy, xx = np.indices((N, N)); RPX = np.hypot(yy - 49.5, xx - 49.5); RK = np.maximum(RPX, 0.3) * PIX


def sigma_map(M, z, A=1.0):
    r200, V200, c = halo(M, z); return A * V200 * s_los(RK / r200, c)


# ---- calibrate on darkskies aperture dispersions (mass-weighted projected sigma^2 within aperture)
db = pd.read_csv(f"{ROOT}/halo_database.csv"); rows = db.drop_duplicates(["halo", "snap", "projection"])
ZMAP = {7: 0.375, 8: 0.25, 9: 0.125, 10: 0.0}
pred, obs = {a: [] for a in (10, 30, 100)}, {a: [] for a in (10, 30, 100)}
for r_ in rows[rows.projection == "x"].itertuples():
    mp, meta = pickle.load(open(r_.cdm_path, "rb")); Sig = np.asarray(mp)[0]; p = meta["props"]
    M = p["Mass_200crit"] * 1e10; z = ZMAP[r_.snap]; s = sigma_map(M, z)
    for a in (10, 30, 100):
        m = RK <= max(a, PIX * 0.75)
        pred[a].append(np.sqrt((Sig[m] * s[m] ** 2).sum() / Sig[m].sum())); obs[a].append(p[f"Aperture_veldisp_{a}_kpc"])
A = float(np.median(np.concatenate([np.array(obs[a]) / np.array(pred[a]) for a in (30, 100)])))
print(f"amplitude A = {A:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
for a, cl in zip((10, 30, 100), ("#2a9d8f", "#e9a03b", "#c0392b")):
    ax[0].scatter(np.array(pred[a]) * A, obs[a], s=8, color=cl, label=f"aperture {a} kpc")
ax[0].plot([200, 1600], [200, 1600], "k", lw=0.8); ax[0].set_xlabel("model σ (NFW Jeans, A×)"); ax[0].set_ylabel("VELOCIraptor aperture σ [km/s]")
ax[0].legend(); ax[0].set_title("calibration on darkskies CDM halos")
for (M, z, ls) in ((10 ** 14.2, 0.0, ":"), (10 ** 14.6, 0.0, "-"), (10 ** 15.0, 0.0, "--"), (10 ** 14.6, 0.375, "-.")):
    ax[1].plot(np.arange(1, 50) * PIX, sigma_map(M, z, A)[49, 50:99], ls, color="k", label=f"log M {np.log10(M):.1f}, z {z}")
ax[1].set_xscale("log"); ax[1].set_xlabel("projected R [kpc]"); ax[1].set_ylabel("σ_los proxy [km/s]"); ax[1].legend(fontsize=8)
ax[1].set_title("proxy profiles")
fig.savefig(f"{HERE}/figures/sigma_v_proxy_calibration.png", dpi=100)

# ---- test: SIDM change vs density, split by sigma_v
fig, ax = plt.subplots(1, 3, figsize=(19, 5.2), constrained_layout=True)
for gi, (suite, f, smax) in enumerate((("bahamas", "pairs.npz", 1.0), ("darkskies", "darkskies_pairs.npz", 0.2))):
    D = np.load(f"{HERE}/data/{f}", allow_pickle=True)
    x, y, xi, sig, sh, zz, Ml = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "redshift", "log10_M200"))
    k = np.where((np.abs(sh).max(1) <= 2) & (sig == smax))[0]
    SV = np.stack([sigma_map(10.0 ** Ml[xi[i]], zz[xi[i]], A) for i in k]).astype(np.float32)
    np.savez_compressed(f"{HERE}/data/sigma_v_proxy_{suite}.npz", pair=k, sigma_v=SV)
    c = x[xi[k]][:, 2:98, 2:98].ravel(); d = (y[k] - x[xi[k]])[:, 2:98, 2:98].ravel(); sv = SV[:, 2:98, 2:98].ravel()
    sel = c > 13.6; c, d, sv = c[sel], d[sel], sv[sel]
    qs = np.quantile(sv, [0, 0.25, 0.5, 0.75, 1.0]); bins = np.linspace(13.6, 15.6, 21); mid = 0.5 * (bins[1:] + bins[:-1])
    for j, cl in enumerate(("#3a6fb0", "#2a9d8f", "#e9a03b", "#c0392b")):
        m = (sv >= qs[j]) & (sv <= qs[j + 1]); idx = np.digitize(c[m], bins) - 1
        med = [np.mean(d[m][idx == b]) if (idx == b).sum() > 200 else np.nan for b in range(len(mid))]
        ax[gi].plot(mid, med, "o-", color=cl, ms=3, label=f"σ_v {qs[j]:.0f}–{qs[j + 1]:.0f} km/s")
    ax[gi].axhline(0, color="0.5", lw=0.7); ax[gi].set_xlabel("CDM pixel log10 Σ"); ax[gi].set_ylabel("mean Δ log (SIDM − CDM)")
    ax[gi].set_title(f"{suite}, σ = {smax}: Δ vs density, split by σ_v proxy (quartiles)"); ax[gi].legend(fontsize=8)
    rate = c + np.log10(sv); rb = np.linspace(np.percentile(rate, 1), np.percentile(rate, 99.99), 25); rm = 0.5 * (rb[1:] + rb[:-1])
    idx = np.digitize(rate, rb) - 1
    ax[2].plot(rm - np.log10(1e3), [np.mean(d[idx == b]) if (idx == b).sum() > 200 else np.nan for b in range(len(rm))], "o-",
               color=("#c0392b" if suite == "bahamas" else "#2a9d8f"), label=f"{suite}, σ = {smax}")
ax[2].axhline(0, color="0.5", lw=0.7); ax[2].set_xlabel("log10 (Σ × σ_v / 1000 km/s)  — scattering-rate proxy")
ax[2].set_ylabel("mean Δ log"); ax[2].legend(); ax[2].set_title("Δ vs rate proxy")
fig.suptitle("does the SIDM change depend on velocity dispersion at fixed density? (σ_v = NFW-Jeans proxy)")
fig.savefig(f"{HERE}/figures/sigma_v_proxy_test.png", dpi=95); print("wrote figures")
