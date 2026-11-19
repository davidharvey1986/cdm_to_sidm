"""Merger (pairs row 1951 = z0.250_291, projection 0): true SIDM vs two-scale converter (darkskies-trained, no CNN)
and the dark-matter LOS velocity maps (mean v - bulk, sigma_los within +-R200) for each model. Central 96x96 px.
usage: python3 plot_merger_conv_los.py -> figures/merger_conv_los_1951.png"""
import os, json, pickle
import numpy as np, matplotlib
from scipy.ndimage import gaussian_filter as G
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import two_scale as T2
HERE = os.path.dirname(os.path.abspath(__file__)); row = 1951
IMG = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas"; V = f"{IMG}/velmaps_v2_imaged"
D = np.load(f"{HERE}/data/pairs.npz", allow_pickle=True)
x, y, xi, sig, zz = (D[k] for k in ("x", "y", "xi", "sigma", "redshift"))
mid, P, z = str(D["match_id"][row]), int(D["projection"][row]), float(zz[row])
SIG = (0.1, 0.3, 1.0); pr = {s: np.where((xi == row) & (sig == s))[0][0] for s in SIG}
cs = slice(2, 98); c = x[row, cs, cs].astype(float)
true = {s: y[pr[s], cs, cs].astype(float) for s in SIG}
J = json.load(open(f"{HERE}/runs/two_scale_darkskies_nocnn/template.json")); conv = {}
for s in SIG:
    sg = np.array([s]); Lc = T2.smooth(c[None]); Sc = c[None] - Lc
    conv[s] = (c[None] + T2.predict_large(Lc, sg, np.array([z]), J["template"]) + (T2.apply_small(Sc, sg, J["small"]) - Sc))[0]
# velocity maps in stamp orientation
O = json.load(open(f"{HERE}/../velmaps/bahamas/orientation.json"))[f"projection_{P}"]
orient = lambda m: (lambda r: r[:, ::-1] if O["flip"] else r)(np.rot90(m, O["rot"]))
meta, _ = pickle.load(open(f"{IMG}/bahamas_matched.pkl", "rb"))
k = np.where((meta["match_id"] == mid) & (meta["projection"] == P))[0][0]
SNAP = {0.0: "032", 0.125: "031", 0.25: "030", 0.375: "029"}
MOD = {0: ("CDM+baryons", "cdm"), 0.1: ("SIDM0.1+baryons", "sidm0.1"), 0.3: ("SIDM0.3+baryons", "sidm0.3"), 1.0: ("SIDM1+baryons", "sidm1")}
vel = {}
for s, (model, key) in MOD.items():
    d = np.load(f"{V}/{model}/snap_{SNAP[z]}/los{O['los']}.npz"); j = np.where(d["fof"] == int(meta[f"fof_{key}"][k]))[0][0]
    M = d["moments"][j].astype(float); W0, W1, W2 = (orient(G(M[4 + q], 1.5)) for q in range(3)); vm = W1 / W0
    vel[s] = (vm[cs, cs], np.sqrt(np.clip(W2 / W0 - vm * vm, 0, None))[cs, cs])
lo, hi = np.percentile(c, [1, 99.9]); kw = dict(origin="lower", interpolation="nearest", extent=[-0.96, 0.96, -0.96, 0.96])
fig, ax = plt.subplots(4, 6, figsize=(25, 16.5), constrained_layout=True)
titles = ["true log10 Σ", "converted log10 Σ (from CDM)", "true Δ = SIDM − CDM", "converted Δ", "mean LOS v − bulk [km/s]", "σ_los, ±R200 [km/s]"]
for i, s in enumerate((0,) + SIG):
    a = ax[i]; lab = "CDM" if s == 0 else f"SIDM{s:g}"
    ims = [a[0].imshow(c if s == 0 else true[s], cmap="magma", vmin=lo, vmax=hi, **kw)]
    if s:
        ims.append(a[1].imshow(conv[s], cmap="magma", vmin=lo, vmax=hi, **kw))
        dd = a[2].imshow(true[s] - c, cmap="RdBu_r", vmin=-.3, vmax=.3, **kw); a[3].imshow(conv[s] - c, cmap="RdBu_r", vmin=-.3, vmax=.3, **kw)
    else:
        for q in (1, 2, 3): a[q].axis("off")
        a[1].text(0.5, 0.5, "input to the converter", ha="center", va="center", transform=a[1].transAxes, fontsize=13, color="0.4")
    vi = a[4].imshow(vel[s][0], cmap="RdBu_r", vmin=-900, vmax=900, **kw); si = a[5].imshow(vel[s][1], cmap="viridis", vmin=300, vmax=1100, **kw)
    a[0].set_ylabel(lab, fontsize=15)
    if i == 0:
        for q, t in enumerate(titles): a[q].set_title(t, fontsize=13)
    if i == 1: a[1].set_title(titles[1], fontsize=13); a[2].set_title(titles[2], fontsize=13); a[3].set_title(titles[3], fontsize=13)
for a_ in ax.ravel(): a_.set_xticks([]); a_.set_yticks([])
fig.colorbar(ims[0], ax=ax[:, 0:2], shrink=0.35, label="log10 Σ total"); fig.colorbar(dd, ax=ax[1:, 2:4], shrink=0.35, label="Δ [dex]")
fig.colorbar(vi, ax=ax[:, 4], shrink=0.35, label="km/s"); fig.colorbar(si, ax=ax[:, 5], shrink=0.35, label="km/s")
fig.suptitle(f"Merger {mid} (projection {P}, z = {z}, log M200 = {float(D['log10_M200'][row]):.2f}): true vs converted, with DM line-of-sight velocity maps", fontsize=15)
out = f"{HERE}/figures/merger_conv_los_{row}.png"; fig.savefig(out, dpi=80); print(out)
