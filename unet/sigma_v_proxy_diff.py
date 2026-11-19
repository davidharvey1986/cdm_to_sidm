"""Velocity-dispersion test without the CDM selection effect: the SIDM-specific change
   d = log Sigma_SIDM(high sigma) - log Sigma_SIDM(low sigma)   of the same cluster (CDM noise cancels),
binned by the CDM pixel density (used only to select pixels; neither map in d is the selection map) and split by
the sigma_v proxy (sigma_v_proxy.py). BAHAMAS: SIDM1 - SIDM0.1; darkskies: SIDM0.2 - SIDM0.01.
usage: python3 sigma_v_proxy_diff.py  -> figures/sigma_v_proxy_diff.png"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(f"{HERE}/sigma_v_proxy.py").read()
exec(src.split("# ---- calibrate")[0]); A = 1.053  # calibrated amplitude from sigma_v_proxy.py
fig, ax = plt.subplots(1, 3, figsize=(19, 5.2), constrained_layout=True)
for gi, (suite, f, (slo, shi)) in enumerate((("bahamas", "pairs.npz", (0.1, 1.0)), ("darkskies", "darkskies_pairs.npz", (0.01, 0.2)))):
    D = np.load(f"{HERE}/data/{f}", allow_pickle=True)
    x, y, xi, sig, sh, zz, Ml = (D[k] for k in ("x", "y", "xi", "sigma", "shift", "redshift", "log10_M200"))
    ok = np.abs(sh).max(1) <= 2; rows = {}
    for i in np.where(ok & np.isin(sig, (slo, shi)))[0]: rows.setdefault(xi[i], {})[float(sig[i])] = i
    cl = [c_ for c_, d_ in rows.items() if slo in d_ and shi in d_]
    hi_ = np.array([rows[c_][shi] for c_ in cl]); lo_ = np.array([rows[c_][slo] for c_ in cl]); cl = np.array(cl)
    SV = np.stack([sigma_map(10.0 ** Ml[c_], zz[c_], A) for c_ in cl])
    c = x[cl][:, 2:98, 2:98].ravel(); d = (y[hi_] - y[lo_])[:, 2:98, 2:98].ravel(); sv = SV[:, 2:98, 2:98].ravel()
    sel = c > 13.6; c, d, sv = c[sel], d[sel], sv[sel]
    qs = np.quantile(sv, [0, 0.25, 0.5, 0.75, 1.0]); bins = np.linspace(13.6, 15.6, 21); mid = 0.5 * (bins[1:] + bins[:-1])
    for j, colr in enumerate(("#3a6fb0", "#2a9d8f", "#e9a03b", "#c0392b")):
        m = (sv >= qs[j]) & (sv <= qs[j + 1]); idx = np.digitize(c[m], bins) - 1
        mean = np.array([d[m][idx == b].mean() if (idx == b).sum() > 200 else np.nan for b in range(len(mid))])
        err = np.array([d[m][idx == b].std() / np.sqrt((idx == b).sum()) if (idx == b).sum() > 200 else np.nan for b in range(len(mid))])
        ax[gi].errorbar(mid, mean, err, fmt="o-", color=colr, ms=3, capsize=0, label=f"σ_v {qs[j]:.0f}–{qs[j + 1]:.0f} km/s")
    ax[gi].axhline(0, color="0.5", lw=0.7); ax[gi].set_xlabel("CDM pixel log10 Σ (selection only)")
    ax[gi].set_ylabel(f"mean log Σ(SIDM {shi:g}) − log Σ(SIDM {slo:g})")
    ax[gi].set_title(f"{suite}: SIDM {shi:g} vs SIDM {slo:g}, same clusters (n={len(cl)})"); ax[gi].legend(fontsize=8)
    rate = c + np.log10(sv); rb = np.linspace(np.percentile(rate, 1), np.percentile(rate, 99.99), 25); rm = 0.5 * (rb[1:] + rb[:-1])
    idx = np.digitize(rate, rb) - 1
    ax[2].plot(rm - 3, [d[idx == b].mean() if (idx == b).sum() > 200 else np.nan for b in range(len(rm))], "o-",
               color=("#c0392b" if suite == "bahamas" else "#2a9d8f"), label=f"{suite}: SIDM {shi:g} − {slo:g}")
ax[2].axhline(0, color="0.5", lw=0.7); ax[2].set_xlabel("log10 (Σ × σ_v / 1000 km/s) — scattering-rate proxy"); ax[2].set_ylabel("mean Δ log")
ax[2].legend(); ax[2].set_title("SIDM-specific change vs rate proxy")
fig.suptitle("SIDM-specific change (two SIDM runs of the same cluster, CDM noise cancels) vs density, split by σ_v proxy")
fig.savefig(f"{HERE}/figures/sigma_v_proxy_diff.png", dpi=95); print("wrote")
