"""Gas and stellar surface-density maps for the BAHAMAS(-SIDM) clusters, from the particle data on COSMA.

Companion to bahamas_vel_maps.py (dark matter): SAME grid, centre, projections and particle selections, plain binning,
so that dark matter (W0 of the velocity maps), gas and stars add up to the total without kernel mismatch.
    deep5 : every particle within +-5 pMpc of the centre along the line of sight (depth of the BAHAMAS mass maps)
    slab  : every particle within +-R200c of the centre along the line of sight

Inputs (COSMA)
  particles : <sims>/<model>/data/particledata_<snap>/*.hdf5 (EAGLE format; the particle set the mass maps were made
              from); PartType0 (gas) and PartType4 (stars): Coordinates [cMpc/h], Mass [1e10 Msun/h]
  catalogue : <catdir>/projection_0/<model>/z_<z>/catalog.dat, as bahamas_vel_maps.py
Output      : <out>/<model>/snap_<snap>/los<k>.npz with maps (Ncl, 8, 100, 100) float32 =
              [Sigma_gas, N_gas, Sigma_star, N_star] for deep5 then slab; Sigma in Msun/pMpc^2, N = particle count.
              Plus fof, id, log10_M200, r200_pMpc, centre_cMpc_h, a, z, los, channels, units.
              Orientation to the stamps as the velocity maps (code/orientation.json).

usage: python3 bahamas_baryon_maps.py --model CDM+baryons --snap 032 --catdir <dir> --out <dir> [--ncl 5]
"""
import os, glob, time, argparse
import numpy as np, h5py
from bahamas_vel_maps import SIMS, ZLAB, NPIX, HALF, DEEP, CHUNK, CellIndex, read_catalog

PT = {"gas": "PartType0", "star": "PartType4"}
CHANNELS = np.array([f"{q}_{s}" for s in ("deep5", "slab") for q in ("Sigma_gas", "N_gas", "Sigma_star", "N_star")])
UNITS = np.array(["Msun/pMpc^2", "count", "Msun/pMpc^2", "count"] * 2)


def load_species(files, pt):
    """positions [pMpc] and masses [Msun] (float32) of one particle type, and the header"""
    P, M, hd = [], [], None
    for fn in files:
        with h5py.File(fn, "r") as f:
            hd = dict(f["Header"].attrs); a, h = float(hd["ExpansionFactor"]), float(hd["HubbleParam"])
            if pt not in f or "Coordinates" not in f[pt]: continue
            n = f[f"{pt}/Coordinates"].shape[0]
            for s in range(0, n, CHUNK):
                P.append((f[f"{pt}/Coordinates"][s:s + CHUNK] * (a / h)).astype(np.float32))
                M.append((f[f"{pt}/Mass"][s:s + CHUNK] * (1e10 / h)).astype(np.float32))
            print(f"  {os.path.basename(fn)} {pt}: {n} particles", flush=True)
    if not P: return np.zeros((0, 3), np.float32), np.zeros(0, np.float32), hd
    return np.concatenate(P), np.concatenate(M), hd


def bin2(u, w, m, A):
    e = np.linspace(-HALF, HALF, NPIX + 1)
    S, _, _ = np.histogram2d(w, u, bins=[e, e], weights=m)  # rows = second image axis, columns = first (as vel maps)
    N, _, _ = np.histogram2d(w, u, bins=[e, e])
    return S / A, N


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--snap", required=True, choices=sorted(ZLAB))
    ap.add_argument("--catdir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ncl", type=int, default=None, help="first N clusters only (testing)")
    ap.add_argument("--sims", default=SIMS)
    a_ = ap.parse_args(); t0 = time.time()
    cat = read_catalog(f"{a_.catdir}/projection_0/{a_.model}/z_{ZLAB[a_.snap]}/catalog.dat", a_.ncl)
    files = sorted(glob.glob(f"{a_.sims}/{a_.model}/data/particledata_{a_.snap}/*.hdf5"))
    print(f"{a_.model} snap {a_.snap} (z {ZLAB[a_.snap]}): {len(cat['fof'])} clusters, {len(files)} particle file(s)", flush=True)
    A = (2 * HALF / NPIX) ** 2; nc = len(cat["fof"])
    res = {k: np.zeros((nc, 8, NPIX, NPIX), np.float32) for k in range(3)}
    for j, (sp, pt) in enumerate(PT.items()):  # one species in memory at a time
        pos, m, hd = load_species(files, pt)
        a, h = float(hd["ExpansionFactor"]), float(hd["HubbleParam"]); box = float(hd["BoxSize"]) * a / h
        pos %= box; idx = CellIndex(pos, box) if len(pos) else None
        print(f"  {sp}: {len(pos)} particles, total {m.sum():.3e} Msun, {time.time() - t0:.0f}s", flush=True)
        for i, (c, r200) in enumerate(zip(cat["pos"] * a / h, cat["r200"])):
            if idx is None: continue
            sel = idx.query(c, DEEP)
            d = pos[sel].astype(np.float64) - c; d -= box * np.round(d / box); ms = m[sel].astype(np.float64)
            for k in range(3):
                i1, i2 = [q for q in range(3) if q != k]
                inplane = (np.abs(d[:, i1]) < HALF) & (np.abs(d[:, i2]) < HALF)
                for o, cut in ((0, DEEP), (4, r200)):
                    s_ = inplane & (np.abs(d[:, k]) < cut)
                    res[k][i, o + 2 * j], res[k][i, o + 2 * j + 1] = bin2(d[s_, i1], d[s_, i2], ms[s_], A)
            if i < 5 or i % 100 == 0:
                cen = res[2][i, 2 * j][45:55, 45:55].mean()
                print(f"  {sp} cluster {i} fof {cat['fof'][i]} logM {cat['log10_M200'][i]:.2f}: central Sigma (z, deep5) "
                      f"{cen:.3e} Msun/pMpc^2, {int(res[2][i, 2 * j + 1].sum())} particles in stamp, {time.time() - t0:.0f}s", flush=True)
        del pos, m, idx
    os.makedirs(f"{a_.out}/{a_.model}/snap_{a_.snap}", exist_ok=True)
    for k in range(3):
        np.savez_compressed(f"{a_.out}/{a_.model}/snap_{a_.snap}/los{k}.npz", maps=res[k], fof=cat["fof"], id=cat["id"],
                            log10_M200=cat["log10_M200"], r200_pMpc=cat["r200"], centre_cMpc_h=cat["pos"], a=a,
                            z=float(ZLAB[a_.snap]), los=k, channels=CHANNELS, units=UNITS,
                            selection=np.array("all particledata gas (PartType0) and stars (PartType4); deep5 = |los| < 5 pMpc, slab = |los| < R200c"))
    print(f"wrote {a_.out}/{a_.model}/snap_{a_.snap}/los{{0,1,2}}.npz in {time.time() - t0:.0f}s", flush=True)
