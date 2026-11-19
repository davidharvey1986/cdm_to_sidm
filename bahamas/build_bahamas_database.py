"""Indexed, halo-matched database of the BAHAMAS cluster stamps.

Inputs
  catalogs/projection_P/<model>/z_<z>/catalog.dat  (copied from
      /srv/astro/projects/dark_skies/bahamas; columns id fof m200 r200 x y z ...,
      log10 M200 [Msun], r200, positions in the 400 Mpc/h box)
  IMG/bahamas.pkl  (all 7 baryonic models, 25816 stamps) and the per-model
      IMG/bahamas_<model>.pkl files; each = (meta dict, images (N,100,100,3)).
      meta['clusterID'] is the catalogue's *fof* column (not the row id).

Matching (per redshift, against the CDM+baryons catalogue): mutual nearest
neighbour in the periodic box with separation < 1.5 Mpc/h and
|dlog M200| < 0.3. The separation distribution has a clean gap between ~1 and
~2 Mpc/h; 90% of matches are within 0.12 Mpc/h (0.43 for CDM_hi). Matching is
done on the full catalogues (M200 > 1e14, vdSIDM+baryons > 10^13.7), not only
the imaged top 300, so a stamp whose partner sits just below the other model's
image cut still gets a match_id. No matching across redshift.

Outputs (in this directory)
  bahamas_stamps.csv    one row per stamp (25816): where it is in bahamas.pkl
                        and in its per-model file, catalogue properties, and
                        match_id = "z<z>_<CDM+baryons fof>" ('' if unmatched)
  bahamas_halo_database.csv  one row per (redshift, projection, model,
                        matched halo) with a stamp in CDM+baryons and/or the
                        model: indices of both stamps (-1 if absent), masses,
                        separation, dlogM
usage: python3 build_bahamas_database.py
"""
import os, pickle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas"
CAT = f"{HERE}/catalogs"
L, MAXSEP, MAXDM, ALIGNED = 400.0, 1.5, 0.3, 0.2
REF = "CDM+baryons"
FILES = {"CDM+baryons": "bahamas_cdm.pkl", "CDM_hi+baryons": "bahamas_cdm_hi.pkl",
         "CDM_low+baryons": "bahamas_cdm_low.pkl", "SIDM0.1+baryons": "bahamas_sidm0.1.pkl",
         "SIDM0.3+baryons": "bahamas_sidm0.3.pkl", "SIDM1+baryons": "bahamas_sidm1.pkl",
         "vdSIDM+baryons": "bahamas_vdsidm.pkl"}
COLS = ["id", "fof", "m200", "r200", "x", "y", "z"]


def catalog(model, z):
    # projection_0 holds the superset catalogue for vdSIDM+baryons; identical otherwise
    c = np.loadtxt(f"{CAT}/projection_0/{model}/z_{z:.3f}/catalog.dat")[:, :7]
    c = pd.DataFrame(c, columns=COLS).astype({"id": int, "fof": int})
    assert c.fof.is_unique
    return c


def load_meta(f):
    m, im = pickle.load(open(f"{IMG}/{f}", "rb"))
    n = len(im)
    del im
    return pd.DataFrame(dict(model=np.asarray(m["sim"]).astype(str),
                             redshift=np.round(np.asarray(m["redshift"], float), 3),
                             projection=np.asarray(m["projection"]).astype(int),
                             fof=np.asarray(m["clusterID"]).astype(int),
                             mass=np.asarray(m["mass"], float),
                             lensing_norm=np.asarray(m["lensing_norm"], float))), n


# ---- stamps: combined file and per-model files
stamps, n = load_meta("bahamas.pkl")
assert len(stamps) == n
stamps["idx_combined"] = np.arange(len(stamps))
key = ["model", "redshift", "projection", "fof"]
assert not stamps.duplicated(key).any()
stamps["model_file"] = stamps.model.map(FILES)
stamps["idx_model_file"] = -1
for model, f in FILES.items():
    m, _ = load_meta(f)
    assert (m.model == model).all(), f
    m["i"] = np.arange(len(m))
    j = stamps.loc[stamps.model == model, key + ["lensing_norm"]].reset_index().merge(
        m[key + ["i", "lensing_norm"]], on=key, how="left", suffixes=("", "_f"))
    assert j.i.notna().all() and len(m) == len(j), f"{f}: stamps differ from bahamas.pkl"
    assert np.allclose(j.lensing_norm, j.lensing_norm_f, rtol=1e-10), f
    stamps.loc[j["index"], "idx_model_file"] = j.i.astype(int).values
    print(f"{f}: {len(m)} stamps, same as bahamas.pkl")

# ---- matching per redshift
cats, matches = {}, []
for z in sorted(stamps.redshift.unique()):
    ref = catalog(REF, z)
    cats[(REF, z)] = ref
    tr = cKDTree(ref[["x", "y", "z"]].values % L, boxsize=L)
    for model in FILES:
        c = catalog(model, z)
        cats[(model, z)] = c
        if model == REF:
            matches.append(pd.DataFrame(dict(model=model, redshift=z, fof=ref.fof, ref_fof=ref.fof,
                                             sep=0.0, dlogM=0.0)))
            continue
        tm = cKDTree(c[["x", "y", "z"]].values % L, boxsize=L)
        d, i = tm.query(ref[["x", "y", "z"]].values % L)  # ref -> model
        _, back = tr.query(c[["x", "y", "z"]].values % L)  # model -> ref
        mutual = back[i] == np.arange(len(ref))
        dm = c.m200.values[i] - ref.m200.values
        ok = mutual & (d < MAXSEP) & (np.abs(dm) < MAXDM)
        matches.append(pd.DataFrame(dict(model=model, redshift=z, fof=c.fof.values[i][ok],
                                         ref_fof=ref.fof.values[ok], sep=d[ok], dlogM=dm[ok])))
        print(f"z={z:.3f} {model:16s} catalogue {len(c):5d}  matched to {REF}: {ok.sum():5d} "
              f"(sep p90 {np.percentile(d[ok], 90):.3f} Mpc/h)")
matches = pd.concat(matches, ignore_index=True)

# ---- attach catalogue properties and match ids to stamps
props = pd.concat([c.assign(model=m, redshift=z) for (m, z), c in cats.items()], ignore_index=True)
stamps = stamps.merge(props.rename(columns={"id": "cat_row"}), on=["model", "redshift", "fof"],
                      how="left", validate="many_to_one")
assert stamps.m200.notna().all() and np.allclose(stamps.m200, stamps.mass, atol=1e-6)
stamps = stamps.merge(matches, on=["model", "redshift", "fof"], how="left", validate="many_to_one")
stamps["match_id"] = np.where(stamps.ref_fof.notna(),
                              "z" + stamps.redshift.map("{:.3f}".format) + "_"
                              + stamps.ref_fof.fillna(-1).astype(int).astype(str), "")
stamps["ref_fof"] = stamps.ref_fof.fillna(-1).astype(int)
stamps["aligned"] = stamps.sep < ALIGNED
stamps = stamps.sort_values("idx_combined")
out_cols = ["idx_combined", "model_file", "idx_model_file", "model", "redshift", "projection",
            "fof", "cat_row", "m200", "r200", "x", "y", "z", "lensing_norm",
            "match_id", "ref_fof", "sep", "dlogM", "aligned"]
stamps[out_cols].rename(columns={"sep": "sep_Mpc_h", "m200": "log10_M200", "x": "x_Mpc_h",
                                 "y": "y_Mpc_h", "z": "z_Mpc_h"}).to_csv(
    f"{HERE}/bahamas_stamps.csv", index=False, float_format="%.6g")

# ---- pair table: CDM+baryons vs each model, per redshift and projection
s = stamps[stamps.match_id != ""]
cdm = s[s.model == REF].set_index(["match_id", "projection"])
rows = []
for model in FILES:
    if model == REF:
        continue
    mm = s[s.model == model].set_index(["match_id", "projection"])
    both = cdm[["idx_combined", "idx_model_file", "fof", "m200"]].join(
        mm[["idx_combined", "idx_model_file", "fof", "m200", "sep", "dlogM", "redshift"]],
        how="outer", lsuffix="_cdm", rsuffix="_model")
    both["model"] = model
    rows.append(both.reset_index())
pairs = pd.concat(rows, ignore_index=True)
pairs["redshift"] = pairs.match_id.str[1:6].astype(float)
# Fill catalogue info for the side without a stamp: the matched halo exists
# but is below that model's image cut. A CDM stamp whose halo has no match in
# the model keeps fof_model = -1 (matched = False).
mt = matches.set_index(["model", "redshift", "ref_fof"])
rf = pairs.match_id.str.split("_").str[1].astype(int)
for side in ("cdm", "model"):
    for i in pairs.index[pairs[f"fof_{side}"].isna()]:
        m = REF if side == "cdm" else pairs.at[i, "model"]
        k = (m, pairs.at[i, "redshift"], rf[i])
        if k not in mt.index:
            continue  # CDM halo with no counterpart in this model
        r = mt.loc[k]
        c = cats[(m, pairs.at[i, "redshift"])].set_index("fof").loc[int(r.fof)]
        pairs.loc[i, [f"fof_{side}", f"m200_{side}"]] = [int(r.fof), c.m200]
        if side == "model":
            pairs.loc[i, ["sep", "dlogM"]] = [r.sep, r.dlogM]
pairs["matched"] = pairs.fof_model.notna()
pairs["aligned"] = pairs.matched & (pairs.sep < ALIGNED)
for c in ("idx_combined_cdm", "idx_model_file_cdm", "idx_combined_model", "idx_model_file_model",
          "fof_cdm", "fof_model"):
    pairs[c] = pairs[c].fillna(-1).astype(int)
pairs["both_stamps"] = (pairs.idx_combined_cdm >= 0) & (pairs.idx_combined_model >= 0)
pairs["cdm_file"] = FILES[REF]
pairs["model_file"] = pairs.model.map(FILES)
pairs = pairs.sort_values(["redshift", "model", "projection", "m200_cdm"],
                          ascending=[True, True, True, False])
pairs[["redshift", "projection", "model", "match_id", "matched", "aligned", "both_stamps",
       "fof_cdm", "fof_model", "m200_cdm", "m200_model", "sep", "dlogM",
       "idx_combined_cdm", "idx_combined_model", "cdm_file", "idx_model_file_cdm",
       "model_file", "idx_model_file_model"]].rename(columns={
           "sep": "sep_Mpc_h", "m200_cdm": "log10_M200_cdm", "m200_model": "log10_M200_model"}).to_csv(
    f"{HERE}/bahamas_halo_database.csv", index=False, float_format="%.6g")

print(f"\n{len(stamps)} stamps, {(stamps.match_id != '').sum()} with a match_id")
print(pairs.groupby(["model"]).agg(rows=("matched", "size"), matched=("matched", "sum"),
                                   both_stamps=("both_stamps", "sum"),
                                   both_aligned=("both_stamps", lambda b: (b & pairs.aligned[b.index]).sum())))
