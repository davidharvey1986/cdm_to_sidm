"""bahamas_matched.pkl: one sample per matched cluster (redshift, projection)
with a (4, 100, 100) image = the total-mass map (channel 0 of the BAHAMAS
stamps) of the same cluster in CDM, SIDM0.1, SIDM0.3 and SIDM1 (all +baryons).

A model's channel is present when its stamp exists and is 'aligned' (centre
< 0.2 Mpc/h from the matched CDM+baryons catalogue halo; see README.md; this
is defined even when the CDM halo has no stamp). Samples with at least 2 of
the 4 channels present are kept; missing channels are NaN and meta['mask']
(N, 4) bool says which channels are present.
Uses bahamas_stamps.csv from build_bahamas_database.py.

Output: IMG/bahamas_matched.pkl = (meta, images)
  images  float64 (N, 4, 100, 100), channel order = meta['channels']
  meta    'mask' (N, 4) bool, 'n_models', match_id, redshift, projection,
          log10_M200 (CDM if present, else mean of the present models), and
          per model (suffix _cdm, _sidm0.1, _sidm0.3, _sidm1): fof,
          log10_M200, sep_Mpc_h (to CDM), idx_combined (row in bahamas.pkl),
          lensing_norm; -1 / NaN where the channel is missing
usage: python3 make_bahamas_matched.py [min_models=2]
"""
import os, sys, pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas"
MODELS = {"cdm": ("CDM+baryons", "bahamas_cdm.pkl"),
          "sidm0.1": ("SIDM0.1+baryons", "bahamas_sidm0.1.pkl"),
          "sidm0.3": ("SIDM0.3+baryons", "bahamas_sidm0.3.pkl"),
          "sidm1": ("SIDM1+baryons", "bahamas_sidm1.pkl")}
TOTAL_MASS = 0  # stamp channels: 0 total mass, 1 X-ray, 2 stellar
MIN_MODELS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
PER = ["fof", "log10_M200", "sep_Mpc_h", "idx_combined", "idx_model_file", "lensing_norm"]

st = pd.read_csv(f"{HERE}/bahamas_stamps.csv", keep_default_na=False)
st = st[(st.match_id != "") & st.aligned.astype(str).eq("True")]
key = ["match_id", "projection"]
sets = None
for tag, (model, _) in MODELS.items():
    s = st[st.model == model][key + PER].rename(columns={c: f"{c}_{tag}" for c in PER})
    sets = s if sets is None else sets.merge(s, on=key, how="outer")
mask = np.stack([sets[f"idx_model_file_{t}"].notna().values for t in MODELS], axis=1)
sets["n_models"] = mask.sum(1)
sets = sets[sets.n_models >= MIN_MODELS].copy()
sets["redshift"] = sets.match_id.str[1:6].astype(float)
masses = sets[[f"log10_M200_{t}" for t in MODELS]]
sets["log10_M200"] = sets["log10_M200_cdm"].fillna(masses.mean(axis=1))
sets = sets.sort_values(["redshift", "projection", "log10_M200"],
                        ascending=[True, True, False]).reset_index(drop=True)
mask = np.stack([sets[f"idx_model_file_{t}"].notna().values for t in MODELS], axis=1)
print(len(sets), "samples; by number of models:", sets.n_models.value_counts().sort_index().to_dict())

images = np.full((len(sets), len(MODELS), 100, 100), np.nan)
for c, (tag, (model, f)) in enumerate(MODELS.items()):
    meta, im = pickle.load(open(f"{IMG}/{f}", "rb"))
    have = mask[:, c]
    i = sets.loc[have, f"idx_model_file_{tag}"].astype(int).values
    # guard: the stamp really is this cluster
    assert (np.asarray(meta["clusterID"]).astype(int)[i] == sets.loc[have, f"fof_{tag}"].values).all()
    assert np.allclose(np.asarray(meta["lensing_norm"])[i], sets.loc[have, f"lensing_norm_{tag}"].values)
    images[have, c] = im[i, :, :, TOTAL_MASS]
    del im
    print("loaded", f, have.sum(), "stamps")

meta = {"channels": [MODELS[t][0] for t in MODELS], "image_quantity": "total mass (channel 0)",
        "missing_channels": "NaN; see mask", "mask": mask, "n_models": sets.n_models.values,
        "match_id": sets.match_id.values, "redshift": sets.redshift.values,
        "projection": sets.projection.values, "log10_M200": sets.log10_M200.values}
for tag in MODELS:
    for c in ("fof", "idx_combined"):
        meta[f"{c}_{tag}"] = sets[f"{c}_{tag}"].fillna(-1).astype(int).values
    for c in ("log10_M200", "sep_Mpc_h", "lensing_norm"):
        meta[f"{c}_{tag}"] = sets[f"{c}_{tag}"].values.astype(float)
out = f"{IMG}/bahamas_matched.pkl"
pickle.dump((meta, images), open(out + ".tmp", "wb"), protocol=4)
os.replace(out + ".tmp", out)
print(out, images.shape)
