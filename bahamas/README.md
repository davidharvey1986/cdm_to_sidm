# BAHAMAS halo-matched stamp database

Built 17 Nov 2026 by `build_bahamas_database.py` from:

- **Images:** `/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images/bahamas/`
  - `bahamas.pkl`: all 7 baryonic models, 25,816 stamps.
  - One `bahamas_<model>.pkl` per model.
  - Each file is `(meta, images)` with images of shape (N, 100, 100, 3).
- **Catalogues:** `/srv/astro/projects/dark_skies/bahamas/projection_P/<model>/z_<z>/catalog.dat`
  - Copied to `catalogs/`.
  - `meta['clusterID']` is the catalogue's **fof** column, not the row `id`. The two differ in many catalogues.

## Files

| File | One row per | Key columns |
|---|---|---|
| `bahamas_stamps.csv` | stamp (25,816) | `idx_combined` (into `bahamas.pkl`), `model_file` + `idx_model_file`, model, redshift, projection, fof, catalogue M200/r200/position, `match_id`, `sep_Mpc_h`, `dlogM`, `aligned` |
| `bahamas_halo_database.csv` | (redshift, projection, model, matched halo) | CDM+baryons vs that model: `idx_*_cdm`, `idx_*_model` (-1 = no stamp), fof and log10 M200 of each, `sep_Mpc_h`, `dlogM`, `matched`, `aligned`, `both_stamps` |

## bahamas_matched.pkl

Built by `make_bahamas_matched.py` and saved in the image folder.

- **Contents.** `(meta, images)`, where `images` is (3594, 4, 100, 100) float64.
- **Channels.** The total-mass map (stamp channel 0) of the same cluster in CDM, SIDM0.1, SIDM0.3 and SIDM1, all +baryons. The order is in `meta['channels']`.
- **Samples.** One per cluster × redshift × projection, with at least 2 of the 4 channels present: 3,126 with all 4, 264 with 3 and 204 with 2. That's about 900 per redshift, with log M200 from 14.02 to 15.48.
- **When a channel is present.** The model's stamp exists and is aligned: its centre is within 0.2 Mpc/h of the matched CDM halo. This is defined even when the CDM halo has no stamp, so 99 samples have no CDM channel.
- **Missing channels.** They are filled with NaN. `meta['mask']` is an (N, 4) bool array of which channels are present, and `meta['n_models']` counts them. To get only the complete sets, use `meta['n_models'] == 4`.
- **`meta` arrays.** `match_id`, `redshift`, `projection` and `log10_M200` (CDM's if present, otherwise the mean over the present models), plus `fof`, `log10_M200`, `sep_Mpc_h`, `idx_combined` (row in `bahamas.pkl`) and `lensing_norm` for each model, with suffix `_cdm`, `_sidm0.1`, `_sidm0.3` or `_sidm1`. These are -1 or NaN for missing channels.

**Checks:**
- In the complete sets, the SIDM vs CDM correlation of the log map has a median of 0.93 (1st percentile 0.85), against 0.75 for random pairs.
- In the partial sets, every pair of present channels has a median correlation of 0.935, with a minimum of 0.80.
- The median central Σ (r < 5 px) relative to CDM is 1.00, 0.996, 0.988 and 0.955 for CDM, SIDM0.1, SIDM0.3 and SIDM1.
- NaN appears exactly where the mask is False.

## How halos are matched

- **Reference.** Matching is done per redshift against the CDM+baryons catalogue.
- **Criteria.** A halo is matched when it is the mutual nearest neighbour in the periodic 400 Mpc/h box, with separation < 1.5 Mpc/h and |Δlog M200| < 0.3.
- **Why these cuts.** The separation distribution has a gap between about 1 and 2 Mpc/h.
- **Which halos.** The full catalogues are matched, not only the imaged top 300, so a halo whose partner falls below the other model's image cut is still matched. The `both_stamps` column says whether both images exist.
- **`match_id`.** It is `z<redshift>_<CDM+baryons fof>`. Stamps with the same `match_id` show the same cluster.

**Use `aligned` for pixel-level work.**
- `aligned` means the two centres are within 0.2 Mpc/h of each other.
- Each stamp is centred on its own model's catalogue centre, so image agreement drops with separation. Correlation of the log total-mass map, CDM vs SIDM1:

| Pairs | Median correlation |
|---|---|
| Separation < 0.1 Mpc/h | 0.93 |
| Separation 0.2–0.3 Mpc/h | 0.81 |
| Separation 0.5–0.75 Mpc/h | 0.56 |
| Random pairs | 0.74 |

- The large-separation matches are mostly mergers whose centre lands on a different clump.

**Unmatched stamps.** 1,338 stamps have no `match_id`:
- 1,158 are vdSIDM+baryons. Its projection_0 sample goes down to 10^13.7, below the 10^14 CDM+baryons catalogue limit.
- The rest are mergers that FoF splits or merges differently between models. One example is fof 4 at z=0, a 10^15 cluster whose centre moves 1.9 Mpc/h between models.

## Caveats

- **vdSIDM+baryons projection_0** has its own, larger stamp selection: 450–466 stamps per redshift, compared with 300 for projections 1 and 2. Its projection_0 catalogue is a superset of the projection_1/2 one, so all three are matched with it.
- **No matching across redshift.** A `match_id` is only defined within one redshift.
- **Dark-matter-only runs are not included.** The CDM, SIDM0.1, SIDM1 and vdSIDM catalogues on bonsai have no local stamps.

## Example

```python
import pickle, pandas as pd
db = pd.read_csv("bahamas_halo_database.csv")
p = db[(db.model == "SIDM1+baryons") & db.both_stamps & db.aligned]
_, cdm = pickle.load(open(IMG + "/bahamas_cdm.pkl", "rb"))
_, sidm = pickle.load(open(IMG + "/bahamas_sidm1.pkl", "rb"))
pairs = cdm[p.idx_model_file_cdm], sidm[p.idx_model_file_model]
```
