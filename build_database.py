"""Build the matched CDM -> SIDM halo database.

One row per (halo, snap, SIDM model, projection). Each row points to the CDM
map (darkskies/CDM, same 20 kpc map product as SIDM; plus the hydro CDM 1 kpc
map) and the SIDM map, with the catalogue-level match diagnostics and a
boolean `matched` flag.

A pair is `matched` when the SIDM halo's minimum-potential centre lies within
MAX_OFFSET_KPC of the CDM one AND |log10(M200c_SIDM / M200c_CDM)| <= MAX_DLOGM.
The continuous diagnostics are kept so the cut can be changed later.

Usage:  python build_database.py [--max-offset-kpc 500] [--max-dlogm 0.1]
Output: halo_database.csv
"""
import os, re, pickle, argparse
import numpy as np
import pandas as pd

BASE = "/Users/davidharvey/Work/swift/eiger_runs/zooms/hydro/images"
SIDM_MODELS = ["SIDM0.01", "SIDM0.05", "SIDM0.1", "SIDM0.2"]
SNAPS = [7, 8, 9, 10]
PROJS = ["x", "y", "z"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halo_database.csv")


def props(path):
    """Halo-catalogue properties used for matching (projection-independent)."""
    _, meta = pickle.load(open(path, "rb"))
    p = meta["props"]
    return dict(redshift=meta["redshift"],
                pos=np.array([p["Xcminpot"], p["Ycminpot"], p["Zcminpot"]]),
                M200c=p["Mass_200crit"], R200c=p["R_200crit"])


def main(max_offset_kpc, max_dlogm):
    rows = []
    for snap in SNAPS:
        halos = sorted({re.sub(r"_[xyz]\.pkl$", "", f)
                        for f in os.listdir(f"{BASE}/darkskies/CDM/{snap}") if f.endswith(".pkl")})
        for h in halos:
            # props are identical across projections, so read the x file once
            cdm = props(f"{BASE}/darkskies/CDM/{snap}/{h}_x.pkl")
            for m in SIDM_MODELS:
                sidm_x = f"{BASE}/darkskies/{m}/{snap}/{h}_x.pkl"
                if os.path.exists(sidm_x):
                    s = props(sidm_x)
                    # catalogue positions are in Mpc
                    offset_kpc = 1e3 * np.linalg.norm(s["pos"] - cdm["pos"])
                    dlogm = np.log10(s["M200c"] / cdm["M200c"])
                    matched = bool(offset_kpc <= max_offset_kpc and abs(dlogm) <= max_dlogm)
                else:
                    s, offset_kpc, dlogm, matched = None, np.nan, np.nan, False
                for pr in PROJS:
                    rows.append(dict(
                        halo=h, snap=snap, redshift=cdm["redshift"],
                        sigma=float(m.replace("SIDM", "")), sidm_model=m, projection=pr,
                        cdm_path=f"{BASE}/darkskies/CDM/{snap}/{h}_{pr}.pkl",
                        sidm_path=f"{BASE}/darkskies/{m}/{snap}/{h}_{pr}.pkl",
                        cdm_hydro_hires_path=f"{BASE}/CDM/{snap}/{h}_{pr}.pkl",
                        M200c_cdm=cdm["M200c"], M200c_sidm=s["M200c"] if s else np.nan,
                        R200c_cdm_kpc=1e3 * cdm["R200c"],
                        offset_kpc=offset_kpc, offset_over_r200c=offset_kpc / (1e3 * cdm["R200c"]),
                        dlogM200c=dlogm, sidm_exists=s is not None, matched=matched))
    df = pd.DataFrame(rows)
    df = add_particle_match(df)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} rows, {df.matched.sum()} matched "
          f"(cuts: offset <= {max_offset_kpc} kpc, |dlogM200c| <= {max_dlogm})")


PM_FILE = os.path.join(os.path.dirname(OUT), "rematch_vs_old.csv")


def add_particle_match(df):
    """Attach the particle-ID re-match (match_halo.py on bonsai, summarised by
    compare_rematch.py) so each row says whether its *existing* images show
    the right halo.

    New columns
      idx_image           VR index the SIDM image was centred on
      pm_idx              VR index chosen by the particle match (-1 = none)
      pm_model_snap       SIDM snapshot at the CDM redshift (differs from `snap`
                          for runs with a shifted output list)
      pm_status           ok / rejected / no_snapshot_at_z / no_cdm_match
      pm_shared, pm_dominance   overlap and winner/runner-up score ratio
      pm_category         readable outcome
      cdm_trace_agrees    CDM image halo == stepwise CDM trace
      image_correct       particle match confirms the existing SIDM image
                          (status ok, same index, same epoch, CDM trace agrees)
      needs_remake        confident new match differs from the existing image
    `matched` (position/mass vs CDM) is unchanged.
    """
    if not os.path.exists(PM_FILE):
        print(f"no {PM_FILE}; skipping particle-match columns")
        return df
    pm = pd.read_csv(PM_FILE).rename(columns={
        "model": "sidm_model", "idx_old": "idx_image", "idx_best": "pm_idx_best",
        "model_snap": "pm_model_snap", "status": "pm_status", "shared": "pm_shared",
        "dominance": "pm_dominance", "category": "pm_category"})
    pm["cdm_trace_agrees"] = ~pm.cdm_changed.astype(bool)
    pm["pm_idx"] = pm.pm_idx_best.where(pm.pm_status == "ok", -1)
    same_epoch = pm.pm_model_snap == pm.snap
    pm["image_correct"] = (pm.pm_status == "ok") & (pm.pm_idx == pm.idx_image) & same_epoch & pm.cdm_trace_agrees
    pm["needs_remake"] = (pm.pm_status == "ok") & ((pm.pm_idx != pm.idx_image) | ~same_epoch)
    cols = ["halo", "snap", "sidm_model", "idx_image", "pm_idx", "pm_model_snap", "pm_status",
            "pm_shared", "pm_dominance", "pm_category", "cdm_trace_agrees",
            "image_correct", "needs_remake"]
    out = df.merge(pm[cols], on=["halo", "snap", "sidm_model"], how="left", validate="many_to_one")
    assert len(out) == len(df) and out.pm_status.notna().all()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-offset-kpc", type=float, default=500.0)
    ap.add_argument("--max-dlogm", type=float, default=0.1)
    a = ap.parse_args()
    main(a.max_offset_kpc, a.max_dlogm)
