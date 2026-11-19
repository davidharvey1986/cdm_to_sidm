"""Collect the halo used for each image (VR ID + key props) from an image tree.

usage: python extract_image_meta.py <image_root> <out.csv> [axis]
Image root layout: <model>/<snap>/<halo>_<axis>.pkl, pickle = (maps, meta).
Plain csv/numpy only so it runs with the system python on bonsai.
"""
import os, sys, csv, pickle
import numpy as np

KEYS = ["ID", "hostHaloID", "Xcminpot", "Ycminpot", "Zcminpot", "Mass_200crit", "R_200crit", "npart"]
root, out = sys.argv[1], sys.argv[2]
axis = sys.argv[3] if len(sys.argv) > 3 else "x"
rows = []
for model in sorted(os.listdir(root)):
    if not os.path.isdir(os.path.join(root, model)):
        continue
    for snap in sorted(os.listdir(os.path.join(root, model))):
        d = os.path.join(root, model, snap)
        for f in sorted(os.listdir(d)):
            if not f.endswith(f"_{axis}.pkl"):
                continue
            maps, meta = pickle.load(open(os.path.join(d, f), "rb"))
            r = dict(model=model, snap=int(snap), halo=f[: -len(f"_{axis}.pkl")],
                     z=meta.get("redshift"), map_shape="x".join(map(str, np.shape(maps))),
                     pix_kpc=meta.get("pixel_size_pkpc"))
            r.update({k: meta["props"].get(k) for k in KEYS})
            rows.append(r)
        print(model, snap, len(rows), flush=True)
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0]))
    w.writeheader()
    w.writerows(rows)
