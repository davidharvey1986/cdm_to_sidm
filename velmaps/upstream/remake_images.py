"""Remake SIDM images centred on the particle-matched halo.

Reruns darkskies_images.get_images with the settings of the original product
but with the VR halo index fixed to the particle match (match_halo.py) instead
of back_trace_halo's choice. Originals are untouched.

Products
  20kpc : as run_all_halos.py        -> 100 px, 2 pMpc stamp, total matter
          output images_rematch/<model>/<snap>/<halo>_<axis>.pkl
  1kpc  : as run_all_halos_byhalo.py -> 1000 px, 1 pMpc stamp, total + stellar
          output images_1kpc_rematch/<model>/<snap>/<halo>_<axis>.pkl
<snap> is the CDM snapshot label (7..10), even when the SIDM run's own
snapshot number differs (model_snap).

The table needs columns halo, snap, model, model_snap, idx_best; an optional
matched_by column is copied into the meta. --outroot writes elsewhere (e.g. a
full regeneration of a product: build_full_table.py makes such a table).

usage (from create_obs/):
  python match_fix/remake_images.py <model>                      # 20kpc, one model
  python match_fix/remake_images.py --product 1kpc --table T.csv # every row of T
"""
import os, sys, csv, time, argparse
import pickle as pkl
import unyt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import darkskies_images as di

RUNS = "/srv/astro/projects/dark_skies/zooms/production_runs"
HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCTS = {
    "20kpc": dict(outroot="images_rematch", table="images_to_remake.csv",
                  kwargs=dict(resolution=100, postage_stamp=2. * unyt.Mpc,
                              total=True, dm=False, xray=False, stellar=False)),
    "1kpc": dict(outroot="images_1kpc_rematch", table="images_1kpc_to_remake.csv",
                 kwargs=dict(resolution=1000, postage_stamp=1. * unyt.Mpc,
                             total=True, dm=False, xray=False, stellar=True)),
}


_raw_hsml = di.generate_smoothing_lengths
_hsml_cache = {}


def _cached_hsml(coordinates, boxsize, **kw):
    """Smoothing lengths are invariant under the axis permutation done for each
    projection, so compute them once per particle set (per halo) and reuse for
    x, y, z. Keyed on particle count + coordinate sum (both permutation
    invariant) + kwargs, so the total-matter and stellar sets stay separate."""
    # per-column sums, sorted: exact under column permutation (a single total
    # sum is not -- float addition order changes with the permutation)
    colsum = np.asarray(coordinates, dtype=np.float64).sum(axis=0)
    key = (len(coordinates), tuple(sorted(colsum.tolist())), tuple(sorted(kw.items())))
    if key not in _hsml_cache:
        t = time.time()
        _hsml_cache[key] = _raw_hsml(coordinates, boxsize, **kw)
        print(f"  smoothing lengths: {len(coordinates)} particles, {time.time()-t:.0f}s", flush=True)
    else:
        print(f"  smoothing lengths: cache hit ({len(coordinates)} particles)", flush=True)
    return _hsml_cache[key].copy()


di.generate_smoothing_lengths = _cached_hsml


def remake(rows, product, outroot=None):
    cfg = PRODUCTS[product]
    outroot = outroot or cfg["outroot"]
    print(f"{product}: {len(rows)} halo/snap/model entries", flush=True)
    t0 = time.time()
    for r in rows:
        model, halo, snap = r["model"], r["halo"], int(float(r["snap"]))
        msnap, idx = int(float(r["model_snap"])), int(float(r["idx_best"]))
        run = f"{RUNS}/{model}/{halo}"
        snapfile = f"{run}/snap/snapshot_{msnap:04d}.hdf5"
        props = f"{run}/velo/veloshot_{msnap:04d}.properties"
        groups = f"{run}/velo/veloshot_{msnap:04d}.catalog_groups"
        outdir = f"{outroot}/{model}/{snap}"
        os.makedirs(outdir, exist_ok=True)
        # bypass back_trace_halo: use the particle-matched index
        di.back_trace_halo = lambda *a, _i=idx, **k: (None, _i)
        _hsml_cache.clear()  # new snapshot: drop the previous halo's lengths
        for ax in "xyz":
            out = f"{outdir}/{halo}_{ax}.pkl"
            if os.path.isfile(out):
                print("already done", out, flush=True)
                continue
            maps, meta = di.get_images(snapfile, props, groups, None, project_axis=ax,
                                       vr_center=True, **cfg["kwargs"])
            meta["vr_index"] = idx
            meta["vr_snapshot"] = msnap
            meta["matched_by"] = r.get("matched_by") or "match_halo.py particle-ID merit (16 Nov 2026)"
            pkl.dump((maps, meta), open(out + ".tmp", "wb"))
            os.replace(out + ".tmp", out)
            print(f"{out} idx={idx} snap={msnap} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", help="only rows for this model (default: all)")
    ap.add_argument("--product", choices=PRODUCTS, default="20kpc")
    ap.add_argument("--table", help="csv of entries (default: the product's list)")
    ap.add_argument("--outroot", help="output tree (default: the product's rematch dir)")
    a = ap.parse_args()
    table = a.table or os.path.join(HERE, PRODUCTS[a.product]["table"])
    rows = [r for r in csv.DictReader(open(table)) if a.model in (None, r["model"])]
    remake(rows, a.product, a.outroot)
