"""Check how well halos match between the hydro CDM zooms and the darkskies
CDM/SIDM runs, using the VELOCIraptor props stored in each image pickle.

Each pickle is (maps, meta); meta['props'] holds the halo-catalogue entry.
Writes matching_check.csv (one row per halo/snapshot/model).
"""
import os, re, pickle, sys
import numpy as np
import pandas as pd

BASE = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images"
MODELS = ["CDM", "darkskies/CDM", "darkskies/SIDM0.01", "darkskies/SIDM0.05",
          "darkskies/SIDM0.1", "darkskies/SIDM0.2"]
SNAPS = ["7", "8", "9", "10"]
PROJ = "x"  # props are per-halo, so one projection is enough
KEYS = ["Xc", "Yc", "Zc", "Xcminpot", "Ycminpot", "Zcminpot", "Mass_200crit",
        "Mass_tot", "R_200crit", "Vmax", "npart", "ID", "hostHaloID"]

rows = []
for snap in SNAPS:
    halos = sorted(re.sub(r"_[xyz]\.pkl$", "", f)
                   for f in os.listdir(f"{BASE}/CDM/{snap}") if f.endswith(f"_{PROJ}.pkl"))
    for h in halos:
        for m in MODELS:
            p = f"{BASE}/{m}/{snap}/{h}_{PROJ}.pkl"
            if not os.path.exists(p):
                rows.append(dict(model=m, snap=int(snap), halo=h, missing=True)); continue
            maps, meta = pickle.load(open(p, "rb"))
            r = dict(model=m, snap=int(snap), halo=h, missing=False,
                     z=meta.get("redshift"), map_shape=str(np.shape(maps)),
                     pix_kpc=meta.get("pixel_size_pkpc"))
            r.update({k: meta["props"].get(k, np.nan) for k in KEYS})
            rows.append(r)
        print(snap, h, flush=True, file=sys.stderr)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(os.path.dirname(__file__), "matching_check.csv"), index=False)
