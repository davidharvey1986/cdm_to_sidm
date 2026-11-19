"""Dark-matter line-of-sight velocity moment maps for the darkskies zooms, matched to the BAHAMAS velocity maps
(velmaps/bahamas/bahamas_vel_maps.py): plain binning of dark-matter particles onto the stamp grid, raw moments stored,
with the same particle selections in both suites.

Grid      : the tot_2mpc_20kpc_nn_100 geometry: 2 pMpc stamp, 100 px of 20 pkpc, VELOCIraptor most-bound-particle centre,
            halo choice from the table (match_fix/full_20kpc_nn100.csv), projections x/y/z as permute_for_projection
            in darkskies_images.py (image columns/rows = the first/second remaining axis in that permutation).
Particles : all high-resolution dark matter (PartType1) of the zoom; velocities = swiftsimio peculiar velocities [km/s],
            relative to the cluster bulk = mean velocity of the dark matter within r < R200c (3-D), as in BAHAMAS.
Selections (along the line of sight, relative to the centre):
    full  : the whole zoom (the 400 cMpc depth of the mass maps)          [darkskies only]
    deep5 : |los| < 5 pMpc  (the depth of the BAHAMAS mass maps)         [both suites]
    slab  : |los| < R200c   (physical, from the VR catalogue)            [both suites]
Output pickle (moments, meta) per halo/axis, moments (12, 100, 100) float32 = [W0, W1, W2, N] for full, deep5, slab:
    W0 = sum m / A [Msun/pMpc^2], W1 = sum m v / A, W2 = sum m v^2 / A, N = particle count (A = pixel area, pMpc^2)
    -> Sigma = W0, <v> = W1/W0, sigma_los = sqrt(W2/W0 - <v>^2)  (smooth W0..W2 first if wanted)
meta: redshift, scale_factor, props (VR row), channels, units, dm particle mass, vr_index / vr_snapshot, R200, bulk velocity.

usage (from create_obs/):
  python match_fix/velmaps/vel_maps.py --table match_fix/velmaps/chunk_00.csv --outroot <dir>
  python match_fix/velmaps/vel_maps.py --table T.csv --outroot <dir> --only 1          # test: first row
"""
import os, csv, time, argparse
import pickle as pkl
import numpy as np
import h5py
from swiftsimio import load

RUNS_DEFAULT = "/srv/astro/projects/dark_skies/zooms/production_runs"  # bonsai path
NPIX, HALF, DEEP = 100, 1.0, 5.0  # pixels, stamp half-width [pMpc], deep5 half-depth [pMpc]
SEL = ("full", "deep5", "slab")
CHANNELS = [f"{w}_{s}" for s in SEL for w in ("W0", "W1", "W2", "N")]
UNITS = ["Msun/pMpc^2", "Msun km/s/pMpc^2", "Msun km^2/s^2/pMpc^2", "count"] * 3
# permute_for_projection (darkskies_images.py): x -> (y, z | x), y -> (x, z | y), z -> (x, y | z)
AXES = {"x": (1, 2, 0), "y": (0, 2, 1), "z": (0, 1, 2)}  # (image column, image row, line of sight)


def vr_centre(props_file, idx, a):
    """VR most-bound-particle centre in physical Mpc, and the halo's catalogue row"""
    with h5py.File(props_file, "r") as P:
        c = np.array([P["Xcmbp"][idx], P["Ycmbp"][idx], P["Zcmbp"][idx]], dtype=np.float64)
        props = {}
        for k in P.keys():
            try: props[k] = P[k][idx]
            except Exception: pass
    return c, props


def moments(u, w, v, m, A):
    e = np.linspace(-HALF, HALF, NPIX + 1); out = []
    for wt in (m, m * v, m * v * v, np.ones_like(v)):
        H, _, _ = np.histogram2d(w, u, bins=[e, e], weights=wt)  # rows = image row axis, columns = image column axis
        out.append(H)
    out[0] /= A; out[1] /= A; out[2] /= A
    return out


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
    dm = s.dark_matter
    pos = np.asarray(dm.coordinates.to("Mpc").value, dtype=np.float64) * a  # comoving -> physical Mpc
    vel = np.asarray(dm.velocities.to("km/s").value, dtype=np.float64)     # peculiar
    m = np.asarray(dm.masses.to("Msun").value, dtype=np.float64)
    d = pos - c; d -= box * np.round(d / box)
    inner = (d * d).sum(1) < r200 ** 2
    vb = (m[inner, None] * vel[inner]).sum(0) / m[inner].sum(); v = vel - vb
    A = (2 * HALF / NPIX) ** 2
    print(f"{model} {snap} {halo}: {len(m)} DM particles ({int(inner.sum())} within R200 = {r200:.2f} pMpc), "
          f"bulk v {np.round(vb).astype(int)} km/s, read {time.time() - t0:.0f}s", flush=True)
    for ax in todo:
        i1, i2, k = AXES[ax]
        inplane = (np.abs(d[:, i1]) < HALF) & (np.abs(d[:, i2]) < HALF)
        out = []
        for sel in (inplane, inplane & (np.abs(d[:, k]) < DEEP), inplane & (np.abs(d[:, k]) < r200)):
            out += moments(d[sel, i1], d[sel, i2], v[sel, k], m[sel], A)
        mom = np.stack(out).astype(np.float32)
        meta = dict(redshift=int(s.metadata.redshift * 1e3) / 1e3, scale_factor=a, props=p, channels=CHANNELS, units=UNITS,
                    pixel_size_pkpc=2e3 * HALF / NPIX, postage_stamp_pMpc=2 * HALF, deep_half_depth_pMpc=DEEP,
                    slab_half_depth_pMpc=r200, dm_particle_mass_Msun=float(np.median(m)), v_bulk_kms=vb,
                    vr_index=idx, vr_snapshot=msnap, axis=ax, image_axes_box=(i1, i2), los_axis_box=k,
                    selection="all high-res DM; full = whole zoom, deep5 = |los| < 5 pMpc, slab = |los| < R200c; bulk = mean v within r < R200c",
                    made_by="velmaps/vel_maps.py (binning, matched to bahamas_vel_maps.py)")
        out_f = f"{outdir}/{halo}_{ax}.pkl"
        pkl.dump((mom, meta), open(out_f + ".tmp", "wb")); os.replace(out_f + ".tmp", out_f)
        cc = slice(47, 53); W0, W1, W2 = (mom[8 + j][cc, cc].sum() for j in range(3))
        print(f"  {out_f}: central sigma_los (slab) {np.sqrt(max(W2 / W0 - (W1 / W0) ** 2, 0)):.0f} km/s, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True); ap.add_argument("--outroot", required=True)
    ap.add_argument("--runs", default=RUNS_DEFAULT); ap.add_argument("--axes", default="xyz")
    ap.add_argument("--only", type=int, default=None, help="process only the first N rows (testing)")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.table)))[: a.only]
    print(f"{len(rows)} rows from {a.table}", flush=True)
    failed = []
    for r in rows:  # one bad row (missing file, read error) must not stop the rest of the chunk
        try:
            one(r, a.runs, a.outroot, a.axes)
        except Exception as e:
            import traceback; traceback.print_exc()
            failed.append(f"{r['model']} {r['snap']} {r['halo']}: {type(e).__name__}: {e}"); print("FAILED", failed[-1], flush=True)
    print(f"done: {len(rows) - len(failed)}/{len(rows)} rows ok", flush=True)
    for f in failed: print("FAILED", f, flush=True)
