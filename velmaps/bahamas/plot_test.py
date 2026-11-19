"""Test BAHAMAS velocity maps (CDM+baryons z=0, 5 clusters) in stamp orientation: stamp total mass, DM surface density,
mean LOS velocity and sigma_los (slab, moments smoothed 1 px). usage: python3 plot_test.py -> test_output/test_maps.png"""
import os, json, pickle
import numpy as np, pandas as pd, matplotlib
from scipy.ndimage import gaussian_filter as G
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); T = f"{HERE}/test_output"; O = json.load(open(f"{HERE}/orientation.json"))
st = pd.read_csv(f"{HERE}/../../bahamas/bahamas_stamps.csv")
_, im = pickle.load(open("/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas/bahamas.pkl", "rb"))
def orient(m, o):
    m = np.rot90(m, o["rot"]); return m[:, ::-1] if o["flip"] else m
o = O["projection_0"]; L = np.load(f"{T}/los{o['los']}.npz")
fig, ax = plt.subplots(5, 4, figsize=(16, 20), constrained_layout=True); kw = dict(origin="lower", extent=[-1, 1, -1, 1])
for i, fof in enumerate(L["fof"]):
    row = st[(st.model == "CDM+baryons") & (st.redshift == 0) & (st.projection == 0) & (st.fof == fof)]
    S = np.asarray(im[int(row.idx_combined.iloc[0])])[..., 0]
    W0, W1, W2 = (orient(G(L["moments"][i, j].astype(float), 1), o) for j in (4, 5, 6))
    vm = W1 / W0; sg = np.sqrt(np.clip(W2 / W0 - vm ** 2, 0, None))
    for j, (img, cm, lo, hi, t) in enumerate(((np.log10(S), "magma", None, None, "stamp: log Σ total"), (np.log10(W0 + 1e11), "magma", None, None, "log Σ DM (±R200)"),
                                               (vm, "RdBu_r", -800, 800, "mean LOS v [km/s]"), (sg, "viridis", 400, 1600, "σ_los [km/s]"))):
        a = ax[i, j].imshow(img, cmap=cm, vmin=lo, vmax=hi, **kw); fig.colorbar(a, ax=ax[i, j], shrink=0.8)
        if i == 0: ax[i, j].set_title(t)
    ax[i, 0].set_ylabel(f"fof {fof}, log M200 {L['log10_M200'][i]:.2f}\npMpc")
fig.suptitle("BAHAMAS CDM+baryons z=0, projection_0: velocity maps from COSMA particle data")
fig.savefig(f"{T}/test_maps.png", dpi=75); print("ok")
