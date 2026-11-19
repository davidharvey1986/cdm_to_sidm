"""Same matching check for images_1kpc (1 kpc, 1000x1000): which VR halo each
image is centred on vs (a) the CDM 1 kpc image (position/mass) and (b) the
particle-ID re-match (rematch.csv). Also compares with the 20 kpc darkskies
images (matching_check.csv) to see whether the two products used the same
halo. Writes check_1kpc.csv."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
m = pd.read_csv(os.path.join(HERE, "meta_1kpc.csv"))
m["idx_1kpc"] = m.ID - 1
print("map shapes:", m.groupby("model").map_shape.unique().to_dict())
print("pixel kpc:", m.groupby("model").pix_kpc.unique().to_dict())

cdm = m[m.model == "CDM"].set_index(["snap", "halo"])
sidm = m[m.model != "CDM"].copy()
c = cdm.loc[list(zip(sidm.snap, sidm.halo))]
pos = ["Xcminpot", "Ycminpot", "Zcminpot"]
sidm["offset_kpc"] = 1e3 * np.sqrt(((sidm[pos].values - c[pos].values) ** 2).sum(1))
sidm["dlogM200c"] = np.log10(sidm.Mass_200crit.values / c.Mass_200crit.values)
sidm["z_cdm"] = c.z.values
sidm["matched"] = (sidm.offset_kpc <= 500) & (sidm.dlogM200c.abs() <= 0.1)

# particle match
pm = pd.read_csv(os.path.join(HERE, "rematch.csv"), dtype={"halo": str})
pm["halo"] = "halo_" + pm.halo.str.zfill(8)
pm = pm.rename(columns={"idx_best": "pm_idx_best", "status": "pm_status", "model_snap": "pm_model_snap"})
sidm = sidm.merge(pm[["halo", "model", "snap", "pm_idx_best", "pm_status", "pm_model_snap"]],
                  on=["halo", "model", "snap"], how="left")
same_epoch = sidm.pm_model_snap == sidm.snap
sidm["image_correct"] = (sidm.pm_status == "ok") & (sidm.pm_idx_best == sidm.idx_1kpc) & same_epoch
sidm["needs_remake"] = (sidm.pm_status == "ok") & ((sidm.pm_idx_best != sidm.idx_1kpc) | ~same_epoch)

# CDM 1kpc vs particle trace, and vs the 20 kpc product
pmc = pm[pm.model == "CDM"].set_index(["snap", "halo"])
cdm = cdm.join(pmc[["pm_idx_best", "pm_status"]])
old = pd.read_csv(os.path.join(HERE, "matching_check.csv"))
old["model"] = old.model.str.replace("darkskies/", "")
old20 = old[old.model.isin(["SIDM0.01", "SIDM0.05", "SIDM0.1", "SIDM0.2"]) |
            (old.model == "CDM")].drop_duplicates(["model", "snap", "halo"])
old20 = old20.assign(idx_20kpc=old20.ID - 1)[["model", "snap", "halo", "idx_20kpc"]]
sidm = sidm.merge(old20, on=["model", "snap", "halo"], how="left")

print(f"\nCDM 1kpc: index differs from particle trace in "
      f"{((cdm.pm_idx_best != cdm.idx_1kpc) & (cdm.pm_status == 'ok')).sum()}/{len(cdm)}")
print("SIDM 1kpc image redshift != CDM redshift:", ((sidm.z - sidm.z_cdm).abs() > 0.005).sum())
print("SIDM 1kpc uses same halo index as 20 kpc image:", (sidm.idx_1kpc == sidm.idx_20kpc).sum(), "/", len(sidm))
print("\nposition/mass matched per model & snap (of 100):")
print(sidm.pivot_table(index="model", columns="snap", values="matched", aggfunc="sum"))
print("\n", pd.crosstab(sidm.matched.map({True: "pos ok", False: "pos BAD"}),
                        [sidm.image_correct, sidm.needs_remake],
                        colnames=["image_correct", "needs_remake"]))
print("\nneeds remake per model:", sidm[sidm.needs_remake].groupby("model").size().to_dict(),
      "total", sidm.needs_remake.sum())
sidm.to_csv(os.path.join(HERE, "check_1kpc.csv"), index=False)
sidm[sidm.needs_remake].rename(columns={"pm_model_snap": "model_snap", "pm_idx_best": "idx_best"})[
    ["halo", "snap", "model", "model_snap", "idx_1kpc", "idx_best"]].to_csv(
    os.path.join(HERE, "images_1kpc_to_remake.csv"), index=False)
