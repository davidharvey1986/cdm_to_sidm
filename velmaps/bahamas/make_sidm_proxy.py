"""Per-pixel SIDM scattering-rate proxy for every BAHAMAS CDM stamp in unet/data/pairs.npz:
    proxy(sigma) = Sigma_DM * sigma_los * (sigma/m)
Sigma_DM [g/cm^2] and sigma_los [km/s] from the CDM run's dark-matter velocity maps (velmaps_v2_imaged), in stamp
orientation (orientation.json); Sigma_DM * (sigma/m) is the projected scattering optical depth tau, so
proxy = tau * sigma_los [km/s]. Moments are smoothed SMOOTH px before dividing (BAHAMAS has ~10-200 DM particles per
pixel). Two line-of-sight selections: slab (+-R200) and deep5 (+-5 pMpc).
Output data/sidm_proxy_bahamas.npz, rows aligned with pairs.npz x:
    Sigma_dm_slab, sigma_los_slab, Sigma_dm_deep5, sigma_los_deep5  (Nc,100,100) float32
    base_slab = Sigma_dm_slab[g/cm^2] * sigma_los_slab[km/s]; proxy(sigma) = base * sigma  (sigma in cm^2/g)
usage: python3 make_sidm_proxy.py
"""
import os, json, pickle
import numpy as np
from scipy.ndimage import gaussian_filter as G

HERE = os.path.dirname(os.path.abspath(__file__)); UNET = os.path.join(HERE, "..", "..", "unet")
IMG = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas"; V = f"{IMG}/velmaps_v2_imaged"
SMOOTH = 1.5; MSUN_PMPC2_TO_G_CM2 = 1.989e33 / 3.0857e24 ** 2
SNAP = {0.0: "032", 0.125: "031", 0.25: "030", 0.375: "029"}
O = json.load(open(f"{HERE}/orientation.json"))
meta, _ = pickle.load(open(f"{IMG}/bahamas_matched.pkl", "rb"))
keep = np.where(meta["mask"][:, 0])[0]
D = np.load(f"{UNET}/data/pairs.npz", allow_pickle=True)
assert (D["match_id"] == meta["match_id"][keep]).all() and (D["projection"] == meta["projection"][keep]).all()
n = len(keep); out = {k: np.full((n, 100, 100), np.nan, np.float32) for k in ("Sigma_dm_slab", "sigma_los_slab", "Sigma_dm_deep5", "sigma_los_deep5")}
cache = {}
for r, k in enumerate(keep):
    z, P, fof = float(meta["redshift"][k]), int(meta["projection"][k]), int(meta["fof_cdm"][k])
    o = O[f"projection_{P}"]; f = f"{V}/CDM+baryons/snap_{SNAP[z]}/los{o['los']}.npz"
    if f not in cache:
        d = np.load(f); cache[f] = (d["moments"], {int(v): i for i, v in enumerate(d["fof"])})
    M, ix = cache[f]
    if fof not in ix: continue
    m = M[ix[fof]].astype(np.float64)
    orient = lambda a: (lambda q: q[:, ::-1] if o["flip"] else q)(np.rot90(a, o["rot"]))
    for off, tag in ((4, "slab"), (0, "deep5")):
        W0, W1, W2 = (G(m[off + q], SMOOTH) for q in range(3)); vm = W1 / W0
        out[f"Sigma_dm_{tag}"][r] = orient(W0); out[f"sigma_los_{tag}"][r] = orient(np.sqrt(np.clip(W2 / W0 - vm * vm, 0, None)))
for tag in ("slab", "deep5"):
    out[f"base_{tag}"] = (out[f"Sigma_dm_{tag}"].astype(np.float64) * MSUN_PMPC2_TO_G_CM2 * out[f"sigma_los_{tag}"]).astype(np.float32)
np.savez_compressed(f"{UNET}/data/sidm_proxy_bahamas.npz", **out, smooth_px=SMOOTH,
                    note="proxy(sigma) = base * sigma[cm^2/g] = Sigma_DM[g/cm^2] * sigma_los[km/s] * sigma/m; CDM run; rows = pairs.npz x")
ok = np.isfinite(out["base_slab"]).all((1, 2))
print(f"{ok.sum()}/{n} CDM stamps with a proxy map; median central Sigma_DM {np.nanmedian(out['Sigma_dm_slab'][:, 49:51, 49:51]) * MSUN_PMPC2_TO_G_CM2:.3f} g/cm^2, "
      f"central sigma_los {np.nanmedian(out['sigma_los_slab'][:, 49:51, 49:51]):.0f} km/s")
