"""Dark-matter line-of-sight velocity moment maps for the BAHAMAS(-SIDM) clusters, from the particle data on COSMA.

Matched to the darkskies velocity maps (velmaps/vel_maps.py --method bin): plain binning of DARK MATTER particles onto
the stamp grid, raw moments stored, with the SAME particle selections in both suites:
    deep5 : every dark-matter particle within +-5 pMpc of the centre along the line of sight
            (the depth of the BAHAMAS mass maps, IDL output_tot_maps.pro: dz = 10 pMpc)
    slab  : every dark-matter particle within +-R200c of the centre along the line of sight
Velocities are relative to the cluster bulk = mean velocity of the dark matter within r < R200c (3-D).

Inputs (COSMA)
  particles : /cosma7/data/dp004/bahamas/sims/L400N1024_WMAP9/<model>/data/particledata_<snap>/*.hdf5
              (EAGLE format, FOF/SO members of every group -- the particle set the mass maps were made from;
              PartType1 Coordinates [cMpc/h], Velocity [km/s a^0.5])
  catalogue : <catdir>/projection_0/<model>/z_<z>/catalog.dat (id fof m200 r200 x y z ...; x,y,z cMpc/h,
              identical for the three projections; m200 log10 Msun, r200 pMpc). All rows mapped; match stamps by fof.
Grid        : 2 pMpc x 2 pMpc, 100 px of 20 pkpc, centred on the catalogue centre (the stamp geometry).
Projections : along each box axis (los = 0, 1, 2 -> x, y, z); image columns = first remaining box axis, rows = second,
              origin lower-left. To the stamp projection_P with code/orientation.json
              (projection_0: los 2; projection_1: los 0; projection_2: los 1 rot90 x3 + flip).
Output      : <out>/<model>/snap_<snap>/los<k>.npz with moments (Ncl, 8, 100, 100) float32 =
              [W0, W1, W2, N] for deep5 then slab; W0 = sum m / A [Msun/pMpc^2], W1 = sum m v / A, W2 = sum m v^2 / A,
              N = particle count. Sigma = W0, <v> = W1/W0, sigma_los = sqrt(W2/W0 - <v>^2) (smooth W0..W2 first if wanted).
              Plus fof, id, log10_M200, r200_pMpc, centre_cMpc_h, v_bulk_kms, a, z, los, m_dm_Msun, channels, units.

usage: python3 bahamas_vel_maps.py --model CDM+baryons --snap 032 --catdir <dir> --out <dir> [--ncl 5]
"""
import os, glob, time, argparse
import numpy as np, h5py

SIMS = "/cosma7/data/dp004/bahamas/sims/L400N1024_WMAP9"
ZLAB = {"029": "0.375", "030": "0.250", "031": "0.125", "032": "0.000"}
NPIX, HALF, DEEP = 100, 1.0, 5.0  # pixels, half-width [pMpc], half-depth of 'deep5' [pMpc]
CELL = 5.0                         # spatial-hash cell [pMpc]
CHUNK = 50_000_000
CHANNELS = np.array(["W0_deep5", "W1_deep5", "W2_deep5", "N_deep5", "W0_slab", "W1_slab", "W2_slab", "N_slab"])
UNITS = np.array(["Msun/pMpc^2", "Msun km/s/pMpc^2", "Msun km^2/s^2/pMpc^2", "count"] * 2)


def read_catalog(path, ncl=None):
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    rows = rows[:ncl] if ncl else rows
    c = np.array([[float(v) for v in r[:7]] for r in rows])
    return dict(id=c[:, 0].astype(int), fof=c[:, 1].astype(int), log10_M200=c[:, 2], r200=c[:, 3], pos=c[:, 4:7])


def load_dm(files):
    """all PartType1 positions [pMpc] and peculiar velocities [km/s] (float32), header"""
    P, V = [], []
    for fn in files:
        with h5py.File(fn, "r") as f:
            hd = dict(f["Header"].attrs); a, h = float(hd["ExpansionFactor"]), float(hd["HubbleParam"])
            n = f["PartType1/Coordinates"].shape[0]
            for s in range(0, n, CHUNK):
                P.append((f["PartType1/Coordinates"][s:s + CHUNK] * (a / h)).astype(np.float32))
                V.append((f["PartType1/Velocity"][s:s + CHUNK] * np.sqrt(a)).astype(np.float32))
                print(f"  read {min(s + CHUNK, n)}/{n} DM particles", flush=True)
    return np.concatenate(P), np.concatenate(V), hd


class CellIndex:
    """particles sorted by a coarse periodic 3-D cell, for fast box queries"""
    def __init__(self, pos, box):
        self.box, self.nc = box, int(np.floor(box / CELL))
        ijk = np.minimum((pos / (box / self.nc)).astype(np.int64), self.nc - 1)
        cid = (ijk[:, 0] * self.nc + ijk[:, 1]) * self.nc + ijk[:, 2]
        self.order = np.argsort(cid, kind="stable"); cs = cid[self.order]
        self.start = np.searchsorted(cs, np.arange(self.nc ** 3)); self.end = np.searchsorted(cs, np.arange(self.nc ** 3), side="right")

    def query(self, c, half):
        """indices of particles in the periodic cube |x - c| < half (per axis), cell-level superset"""
        w = self.box / self.nc
        rng = [np.arange(int(np.floor((c[k] - half) / w)), int(np.floor((c[k] + half) / w)) + 1) % self.nc for k in range(3)]
        I, J, K = np.meshgrid(*rng, indexing="ij"); cells = np.unique(((I * self.nc + J) * self.nc + K).ravel())
        return np.concatenate([self.order[self.start[q]:self.end[q]] for q in cells])


def moments(u, w, v, m, A):
    e = np.linspace(-HALF, HALF, NPIX + 1); out = []
    for wt in (m, m * v, m * v * v, np.ones_like(v)):
        H, _, _ = np.histogram2d(w, u, bins=[e, e], weights=wt)  # rows = second image axis, columns = first
        out.append(H)
    out[0] /= A; out[1] /= A; out[2] /= A
    return out


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
    pos, vel, hd = load_dm(files)
    a, h = float(hd["ExpansionFactor"]), float(hd["HubbleParam"]); box = float(hd["BoxSize"]) * a / h
    mdm = float(hd["MassTable"][1]) * 1e10 / h; A = (2 * HALF / NPIX) ** 2
    pos %= box
    idx = CellIndex(pos, box)
    print(f"  a={a:.4f} box={box:.1f} pMpc, m_dm={mdm:.3e} Msun, {len(pos)} DM particles indexed, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(f"{a_.out}/{a_.model}/snap_{a_.snap}", exist_ok=True)
    nc = len(cat["fof"])
    res = {k: np.zeros((nc, 8, NPIX, NPIX), np.float32) for k in range(3)}; vb = np.zeros((nc, 3))
    for i, (c, r200) in enumerate(zip(cat["pos"] * a / h, cat["r200"])):
        sel = idx.query(c, DEEP)
        d = pos[sel].astype(np.float64) - c; d -= box * np.round(d / box)
        v = vel[sel].astype(np.float64); r3 = np.sqrt((d * d).sum(1))
        inner = r3 < r200
        if not inner.any(): continue
        vb[i] = v[inner].mean(0); v -= vb[i]; m = np.full(len(sel), mdm)
        for k in range(3):
            i1, i2 = [j for j in range(3) if j != k]
            inplane = (np.abs(d[:, i1]) < HALF) & (np.abs(d[:, i2]) < HALF)
            deep = inplane & (np.abs(d[:, k]) < DEEP); slab = inplane & (np.abs(d[:, k]) < r200)
            res[k][i, :4] = moments(d[deep, i1], d[deep, i2], v[deep, k], m[deep], A)
            res[k][i, 4:] = moments(d[slab, i1], d[slab, i2], v[slab, k], m[slab], A)
        if i < 5 or i % 100 == 0:
            W0, W1, W2 = (res[2][i, j][47:53, 47:53].sum() for j in (4, 5, 6))
            print(f"  cluster {i} fof {cat['fof'][i]} logM {cat['log10_M200'][i]:.2f}: {int(inner.sum())} DM within R200, "
                  f"central sigma_los(z, slab) {np.sqrt(max(W2 / W0 - (W1 / W0) ** 2, 0)):.0f} km/s, {time.time() - t0:.0f}s", flush=True)
    for k in range(3):
        np.savez_compressed(f"{a_.out}/{a_.model}/snap_{a_.snap}/los{k}.npz", moments=res[k], fof=cat["fof"], id=cat["id"],
                            log10_M200=cat["log10_M200"], r200_pMpc=cat["r200"], centre_cMpc_h=cat["pos"], v_bulk_kms=vb,
                            a=a, z=float(ZLAB[a_.snap]), los=k, m_dm_Msun=mdm, channels=CHANNELS, units=UNITS,
                            selection=np.array("all particledata DM; deep5 = |los| < 5 pMpc, slab = |los| < R200c; bulk = mean v within r < R200c"))
    print(f"wrote {a_.out}/{a_.model}/snap_{a_.snap}/los{{0,1,2}}.npz in {time.time() - t0:.0f}s", flush=True)
