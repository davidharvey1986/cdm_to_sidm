"""Keep only the clusters that have stamps (imaged_fofs.json, from bahamas_stamps.csv) from maps_v2 -> maps_v2_imaged,
so the transfer is small. usage (COSMA, in bahamas_velmaps/): python3 code/subset_imaged.py"""
import os, json, glob
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
keep = json.load(open(f"{R}/code/imaged_fofs.json"))
for key, fofs in keep.items():
    for f in sorted(glob.glob(f"{R}/maps_v2/{key}/los*.npz")):
        d = dict(np.load(f)); sel = np.isin(d["fof"], fofs)
        for k in ("moments", "fof", "id", "log10_M200", "r200_pMpc", "centre_cMpc_h", "v_bulk_kms"):
            d[k] = d[k][sel]
        o = f.replace("/maps_v2/", "/maps_v2_imaged/"); os.makedirs(os.path.dirname(o), exist_ok=True)
        np.savez_compressed(o, **d)
        print(o, int(sel.sum()), "of", len(sel), "clusters; missing:", sorted(set(fofs) - set(d["fof"].tolist()))[:5], flush=True)
