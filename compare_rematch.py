"""Compare particle-ID re-matching (rematch.csv, from match_halo.py on bonsai)
with the halo indices actually used for the existing images (VR ID stored in
each image's meta, collected in matching_check.csv)."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
new = pd.read_csv(os.path.join(HERE, "rematch.csv"), dtype={"halo": str})
new["halo"] = "halo_" + new.halo.str.zfill(8)
old = pd.read_csv(os.path.join(HERE, "matching_check.csv"))
old = old[old.model != "CDM"].copy()                       # darkskies/CDM == hydro CDM
old["model"] = old.model.str.replace("darkskies/", "")
old["idx_old"] = old.ID - 1
df = new.merge(old[["model", "snap", "halo", "idx_old"]], on=["model", "snap", "halo"])

# old position/mass flag (vs the OLD CDM pick) from the database build
db = pd.read_csv(os.path.join(HERE, "halo_database.csv")).drop_duplicates(["halo", "snap", "sidm_model"])
df = df.merge(db[["halo", "snap", "sidm_model", "matched"]].rename(
    columns={"sidm_model": "model", "matched": "old_matched"}), on=["model", "snap", "halo"], how="left")

df["changed"] = df.idx_best != df.idx_old
df["new_good"] = (df.offset_kpc <= 500) & (df.dlogM200c.abs() <= 0.1)
cdm = df[df.model == "CDM"]
sidm = df[df.model != "CDM"].merge(
    cdm[["halo", "snap", "changed"]].rename(columns={"changed": "cdm_changed"}), on=["halo", "snap"])

print(f"CDM reference halo index changed vs images: {cdm.changed.sum()}/{len(cdm)}")
print(cdm.groupby("snap").changed.sum().to_string(), "\n")
print("SIDM status:", sidm.status.value_counts().to_dict())
print("SIDM index changed vs images:", sidm.changed.sum(), "/", len(sidm))
print("\nold-flag vs new particle match (rows):")
print(pd.crosstab(sidm.old_matched.map({True: "old ok", False: "old BAD"}),
                  [sidm.status, sidm.new_good.map({True: "pos ok", False: "pos bad"})]))
bad = sidm[~sidm.old_matched.astype(bool)]
print(f"\nof {len(bad)} previously mismatched rows: index changed {bad.changed.sum()}, "
      f"CDM ref changed {bad.cdm_changed.sum()}, new match accepted {(bad.status=='ok').sum()}, "
      f"accepted & position-consistent {((bad.status=='ok') & bad.new_good).sum()}")
cols = ["halo", "snap", "model", "idx_old", "idx_best", "status", "shared", "dominance",
        "offset_kpc", "dlogM200c", "cdm_changed"]
print(bad[cols].round(2).sort_values(["halo", "snap", "model"]).to_string(index=False))


def category(r):
    if r.status == "no_snapshot_at_z":
        return "SIDM run has no snapshot at this z (image from wrong epoch)"
    if r.status == "no_cdm_match":
        return "CDM VR particle file inconsistent at this snap"
    if r.status == "ok":
        return "fixed: confident match, position-consistent" if r.new_good else "confident match but offset/mass off"
    if r.shared < 50:
        return "no counterpart in SIDM catalogue"
    return "ambiguous (two candidates)" + (", best is position-consistent" if r.new_good else "")


bad = bad.assign(category=bad.apply(category, axis=1))
print("\nOUTCOME for the previously mismatched rows:")
print(bad.category.value_counts().to_string())
sidm = sidm.assign(category=sidm.apply(category, axis=1))
print("\nOUTCOME for all SIDM rows:")
print(sidm.category.value_counts().to_string())

sidm.to_csv(os.path.join(HERE, "rematch_vs_old.csv"), index=False)
