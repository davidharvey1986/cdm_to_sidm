"""Gas and stellar surface-density maps for the darkskies zooms, matched to the dark-matter velocity maps (vel_maps.py)
and to the BAHAMAS baryon maps (velmaps/bahamas/bahamas_baryon_maps.py): SAME grid, centre, projections, chunk tables and
line-of-sight selections, plain binning, so dark matter (W0 of the velocity maps), gas and stars add up without kernel
mismatch. Needed to test the DM -> stars -> gas chain (trained on BAHAMAS) on darkskies.

Grid      : 2 pMpc stamp, 100 px of 20 pkpc, VELOCIraptor most-bound-particle centre, projections x/y/z as vel_maps.py.
Particles : PartType0 (gas) and PartType4 (stars) of the zoom, swiftsimio masses [Msun].
Selections: deep5 |los| < 5 pMpc (as the BAHAMAS baryon maps and DM pairs), slab |los| < R200c.
Output pickle (maps, meta) per halo/axis at <outroot>/<model>/<snap>/<halo>_<axis>.pkl, maps (8, 100, 100) float32 =
    [Sigma_gas, N_gas, Sigma_star, N_star] for deep5 then slab; Sigma in Msun/pMpc^2, N = particle count.

usage (from create_obs/, same chunk tables as the velocity maps):
  python match_fix/velmaps/baryon_maps.py --table match_fix/velmaps/chunk_00.csv --outroot <dir>
  python match_fix/velmaps/baryon_maps.py --table T.csv --outroot <dir> --only 1          # test: first row
"""
import os, csv, time, argparse
import pickle as pkl
import numpy as np
from swiftsimio import load
from vel_maps import RUNS_DEFAULT, NPIX, HALF, DEEP, AXES, vr_centre

CHANNELS = [f"{w}_{s}" for s in ("deep5", "slab") for w in ("Sigma_gas", "N_gas", "Sigma_star", "N_star")]
UNITS = ["Msun/pMpc^2", "count"] * 4


def particles(part, a):
    try:
        pos = np.asarray(part.coordinates.to("Mpc").value, dtype=np.float64) * a  # comoving -> physical Mpc
        m = np.asarray(part.masses.to("Msun").value, dtype=np.float64)
    except Exception:  # no particles of this type
        pos, m = np.zeros((0, 3)), np.zeros(0)
    return pos, m


def bin2(u, w, m, A):
    e = np.linspace(-HALF, HALF, NPIX + 1)
    S, _, _ = np.histogram2d(w, u, bins=[e, e], weights=m)  # rows = image row axis, columns = image column axis
    N, _, _ = np.histogram2d(w, u, bins=[e, e])
    return [S / A, N]


def one(row, runs, outroot, axes):
    model, halo, snap = row["model"], row["halo"], int(float(row["snap"]))
    msnap, idx = int(float(row["model_snap"])), int(float(row["idx_best"]))
    run = f"{runs}/{model}/{halo}"
    snapfile = f"{run}/snap/snapshot_{msnap:04d}.hdf5"; props = f"{run}/velo/veloshot_{msnap:04d}.properties"
    outdir = f"{outroot}/{model}/{snap}"; os.makedirs(outdir, exist_ok=True)
    todo = [ax for ax in axes if not os.path.isfile(f"{outdir}/{halo}_{ax}.pkl")]
    if not todo:
        print("already done", model, snap, halo, flush=True); return
    t0 = time.time()
    s = load(snapfile); a = float(s.metadata.a)
    box = np.asarray(s.metadata.boxsize.to("Mpc").value, dtype=np.float64) * a  # pMpc
    c, p = vr_centre(props, idx, a); r200 = float(p["R_200crit"])  # VR is physical Mpc
    A = (2 * HALF / NPIX) ** 2
    parts = {}
    for name, part in (("gas", s.gas), ("star", s.stars)):
        pos, m = particles(part, a); d = pos - c; d -= box * np.round(d / box); parts[name] = (d, m)
    print(f"{model} {snap} {halo}: {len(parts['gas'][1])} gas, {len(parts['star'][1])} star particles, read {time.time() - t0:.0f}s", flush=True)
    for ax in todo:
        i1, i2, k = AXES[ax]; out = {}
        for name, (d, m) in parts.items():
            inplane = (np.abs(d[:, i1]) < HALF) & (np.abs(d[:, i2]) < HALF)
            for sel_name, sel in (("deep5", inplane & (np.abs(d[:, k]) < DEEP)), ("slab", inplane & (np.abs(d[:, k]) < r200))):
                out[sel_name, name] = bin2(d[sel, i1], d[sel, i2], m[sel], A)
        maps = np.stack([x for s_ in ("deep5", "slab") for n in ("gas", "star") for x in out[s_, n]]).astype(np.float32)
        meta = dict(redshift=int(s.metadata.redshift * 1e3) / 1e3, scale_factor=a, props=p, channels=CHANNELS, units=UNITS,
                    pixel_size_pkpc=2e3 * HALF / NPIX, postage_stamp_pMpc=2 * HALF, deep_half_depth_pMpc=DEEP,
                    slab_half_depth_pMpc=r200, vr_index=idx, vr_snapshot=msnap, axis=ax, image_axes_box=(i1, i2), los_axis_box=k,
                    selection="all gas (PartType0) and stars (PartType4); deep5 = |los| < 5 pMpc, slab = |los| < R200c",
                    made_by="velmaps/baryon_maps.py (binning, matched to vel_maps.py and bahamas_baryon_maps.py)")
        out_f = f"{outdir}/{halo}_{ax}.pkl"
        pkl.dump((maps, meta), open(out_f + ".tmp", "wb")); os.replace(out_f + ".tmp", out_f)
        print(f"  {out_f}: M_gas, M_star (deep5, < 100 kpc) {maps[0][45:55, 45:55].sum() * A:.3g}, {maps[2][45:55, 45:55].sum() * A:.3g} Msun, "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, nargs="+"); ap.add_argument("--outroot", required=True)
    ap.add_argument("--runs", default=RUNS_DEFAULT); ap.add_argument("--axes", default="xyz")
    ap.add_argument("--models", default=None, help="comma-separated models to keep (default all)")
    ap.add_argument("--only", type=int, default=None, help="process only the first N rows (testing)")
    a = ap.parse_args()
    rows = [r for t in a.table for r in csv.DictReader(open(t))]
    if a.models: rows = [r for r in rows if r["model"] in a.models.split(",")]
    rows = rows[: a.only]
    print(f"{len(rows)} rows from {len(a.table)} tables", flush=True)
    failed = []
    for r in rows:  # one bad row must not stop the rest
        try:
            one(r, a.runs, a.outroot, a.axes)
        except Exception as e:
            import traceback; traceback.print_exc()
            failed.append(f"{r['model']} {r['snap']} {r['halo']}: {type(e).__name__}: {e}"); print("FAILED", failed[-1], flush=True)
    print(f"done: {len(rows) - len(failed)}/{len(rows)} rows ok", flush=True)
    for f in failed: print("FAILED", f, flush=True)
