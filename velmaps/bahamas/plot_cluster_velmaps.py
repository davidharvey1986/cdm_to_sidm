"""BAHAMAS velocity maps of one cluster in CDM and the three SIDM models, in stamp orientation:
stamp total mass, DM surface density (slab), mean LOS velocity, sigma_los (slab and deep5), and radial sigma profiles.
Moments smoothed by SMOOTH px before sigma. Cluster given by match_id and projection (bahamas_matched.pkl).
usage: python3 plot_cluster_velmaps.py [match_id z0.250_291] [projection 0]  -> figures/velmaps_<match_id>_p<P>.png"""
import os, sys, json, pickle
import numpy as np, matplotlib
from scipy.ndimage import gaussian_filter as G
matplotlib.use("Agg"); import matplotlib.pyplot as plt
IMG = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas"; V = f"{IMG}/velmaps_v2_imaged"
HERE = os.path.dirname(os.path.abspath(__file__)); SMOOTH = 1.5
mid = sys.argv[1] if len(sys.argv) > 1 else "z0.250_291"; P = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SNAP = {0.0: "032", 0.125: "031", 0.25: "030", 0.375: "029"}
O = json.load(open(f"{HERE}/orientation.json"))[f"projection_{P}"]
meta, _ = pickle.load(open(f"{IMG}/bahamas_matched.pkl", "rb"))
k = np.where((meta["match_id"] == mid) & (meta["projection"] == P))[0][0]; z = float(meta["redshift"][k])
_, allim = pickle.load(open(f"{IMG}/bahamas.pkl", "rb"))
orient = lambda m: (lambda r: r[:, ::-1] if O["flip"] else r)(np.rot90(m, O["rot"]))
MODELS = [("CDM", "CDM+baryons", "cdm"), ("SIDM0.1", "SIDM0.1+baryons", "sidm0.1"), ("SIDM0.3", "SIDM0.3+baryons", "sidm0.3"), ("SIDM1", "SIDM1+baryons", "sidm1")]
fig, ax = plt.subplots(4, 6, figsize=(28, 19), constrained_layout=True); kw = dict(origin="lower", extent=[-1, 1, -1, 1])
rr = np.hypot(*np.indices((100, 100)) - 49.5).astype(int); cnt = np.bincount(rr.ravel()); R = np.arange(len(cnt)) * 20
cols = {"CDM": "k", "SIDM0.1": "#2a9d8f", "SIDM0.3": "#e9a03b", "SIDM1": "#c0392b"}
for i, (lab, model, key) in enumerate(MODELS):
    fof = int(meta[f"fof_{key}"][k]); idx = int(meta[f"idx_combined_{key}"][k])
    if fof < 0: continue
    d = np.load(f"{V}/{model}/snap_{SNAP[z]}/los{O['los']}.npz"); j = np.where(d["fof"] == fof)[0][0]
    M = d["moments"][j].astype(float)
    def sig(o):
        W0, W1, W2 = (orient(G(M[o + q], SMOOTH)) for q in range(3)); vm = W1 / W0
        return W0, vm, np.sqrt(np.clip(W2 / W0 - vm * vm, 0, None))
    W0s, vms, sgs = sig(4); _, _, sgd = sig(0)
    S = np.asarray(allim[idx])[..., 0]
    panels = [(np.log10(S), "magma", None, None, "stamp: log Σ total"), (np.log10(W0s + 1e11), "magma", None, None, "log Σ DM (±R200)"),
              (vms, "RdBu_r", -900, 900, "mean LOS v − bulk [km/s]"), (sgs, "viridis", 300, 1100, "σ_los, ±R200 [km/s]"),
              (sgd, "viridis", 300, 1100, "σ_los, ±5 pMpc [km/s]")]
    for c, (img, cm, lo, hi, t) in enumerate(panels):
        im = ax[i, c].imshow(img, cmap=cm, vmin=lo, vmax=hi, **kw); fig.colorbar(im, ax=ax[i, c], shrink=0.75)
        if i == 0: ax[i, c].set_title(t, fontsize=12)
    ax[i, 0].set_ylabel(f"{lab}  (fof {fof})\npMpc", fontsize=12)
    for a_ in range(4):
        ax[a_, 5].plot(R, np.bincount(rr.ravel(), np.nan_to_num(sgs).ravel()) / cnt, color=cols[lab], lw=2.4 if a_ == i else 1.0,
                       alpha=1 if a_ == i else 0.5, label=lab)
for a_ in range(4):
    ax[a_, 5].set_xscale("log"); ax[a_, 5].set_xlabel("R [kpc]"); ax[a_, 5].set_ylabel("azimuthal σ_los (±R200) [km/s]")
ax[0, 5].legend(); ax[0, 5].set_title("radial σ_los profiles", fontsize=12)
fig.suptitle(f"BAHAMAS {mid}, projection_{P}, z = {z}, log M200 = {meta['log10_M200'][k]:.2f}: dark-matter velocity maps "
             f"(moments smoothed {SMOOTH} px)", fontsize=15)
out = f"{HERE}/../../unet/figures/velmaps_{mid}_p{P}.png"; fig.savefig(out, dpi=70); print(os.path.abspath(out))
