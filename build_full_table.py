"""Table of every (model, snap, halo) in a stamp catalogue with the VR index
and snapshot its stamps are centred on, read from the existing x-projection
meta: remade stamps carry vr_index/vr_snapshot, the rest use props ID-1 and
the CDM snapshot number. Used to regenerate a whole product with new settings
while keeping the halo choice. System python only (csv, pickle).

usage: python3 build_full_table.py <catalogue_root> <out.csv>
"""
import os, sys, csv, pickle

root, out = sys.argv[1], sys.argv[2]
rows = []
for model in sorted(os.listdir(root)):
    if not os.path.isdir(os.path.join(root, model)):
        continue
    for snap in sorted(os.listdir(os.path.join(root, model)), key=int):
        d = os.path.join(root, model, snap)
        for f in sorted(os.listdir(d)):
            if not f.endswith("_x.pkl"):
                continue
            meta = pickle.load(open(os.path.join(d, f), "rb"))[1]
            remade = "vr_index" in meta
            idx = meta["vr_index"] if remade else int(meta["props"]["ID"]) - 1
            rows.append(dict(halo=f[:-6], snap=int(snap), model=model,
                             model_snap=meta.get("vr_snapshot", int(snap)), idx_best=idx,
                             matched_by=meta["matched_by"] if remade else
                             "original pipeline halo index (kept)"))
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "entries;", sum(r["matched_by"] != "original pipeline halo index (kept)" for r in rows), "remade")
