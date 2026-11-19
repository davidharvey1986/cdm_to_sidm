"""Particle-ID halo matching between CDM and SIDM zoom runs (fix for
validate_halo.back_trace_halo).

Changes relative to back_trace_halo:
  * reads the VELOCIraptor catalogue directly (h5py) and scores *every*
    structure in one vectorised pass -- no maxchecks / nchecks cut-off, no bare
    except;
  * ranks candidates by the merit  N_shared^2 / (N_ref * N_cand)  (fraction of
    the reference recovered x purity of the candidate) instead of the raw count;
  * substructure particles are added to their host, so a host is not penalised
    for a split-off subhalo;
  * never falls back to index 0 or to "first halo with any overlap": a failed
    match returns index -1 together with the diagnostics;
  * matches like-with-like in time: the CDM halo is traced back one snapshot
    at a time from z=0 (main progenitor = most shared particles), then the SIDM
    halo is matched to that CDM halo in the SIDM snapshot at the same redshift
    (SIDM output lists are not always the same as CDM's).
"""
import os
import numpy as np
import h5py

RUNS = "/srv/astro/projects/dark_skies/zooms/production_runs"


class Catalogue:
    """Bound particle IDs per VELOCIraptor structure, grouped by top-level host."""

    def __init__(self, velo_stem):
        props = velo_stem + ".properties"
        groups = velo_stem + ".catalog_groups"
        parts = velo_stem + ".catalog_particles"
        if not os.path.isfile(props):  # multi-file catalogues
            props, groups, parts = props + ".0", groups + ".0", parts + ".0"
        with h5py.File(groups, "r") as g:
            self.offset = g["Offset"][:].astype(np.int64)
        with h5py.File(parts, "r") as f:
            self.pids = f["Particle_IDs"][:]
        with h5py.File(props, "r") as p:
            ids = p["ID"][:].astype(np.int64)
            host = p["hostHaloID"][:].astype(np.int64)
            self.props = {k: p[k][:] for k in
                          ("Xcminpot", "Ycminpot", "Zcminpot", "Xcmbp", "Ycmbp",
                           "Zcmbp", "Mass_200crit", "R_200crit")}
        self.nbound = np.diff(np.append(self.offset, len(self.pids)))
        # index of each structure's top-level host (itself for field halos);
        # VR IDs are index+1 for single-file catalogues
        assert (ids == np.arange(len(ids)) + 1).all(), "VR IDs are not index+1"
        self.host_idx = np.where(host == -1, np.arange(len(ids)), host - 1)
        self.n_host = np.bincount(self.host_idx, weights=self.nbound,
                                  minlength=len(ids)).astype(np.int64)

    def shared_per_host(self, ref_ids):
        """Number of ref particles bound to each host (incl. its subhalos)."""
        hit = np.isin(self.pids, ref_ids, assume_unique=False)
        cs = np.concatenate([[0], np.cumsum(hit)])
        shared = cs[self.offset + self.nbound] - cs[self.offset]
        return np.bincount(self.host_idx, weights=shared,
                           minlength=len(self.offset)).astype(np.int64)

    def host_ids(self, idx):
        """Bound particle IDs of host `idx` including its substructure."""
        members = np.flatnonzero(self.host_idx == idx)
        return np.concatenate([self.pids[self.offset[m]:self.offset[m] + self.nbound[m]]
                               for m in members])


def best_match(cat, ref_ids, by="merit", min_shared=50, min_dominance=5.0):
    """Return (idx, diagnostics). idx = -1 if no acceptable match.

    Absolute overlap is a poor test across DM models: VR structures here are
    small cores and SIDM scattering reshuffles core particles, so only ~10-25%
    of IDs carry over even for the right halo. Instead require the winner to
    dominate the runner-up (score >= min_dominance x 2nd best) and share at
    least min_shared particles.

    by='merit'  : cross-model match at fixed time (SIDM <-> CDM).
    by='shared' : main-progenitor trace in time (most shared particles).
    """
    n_ref = len(ref_ids)
    shared = cat.shared_per_host(ref_ids)
    is_host = cat.host_idx == np.arange(len(shared))
    n_cand = np.maximum(cat.n_host, 1)
    merit = np.where(is_host, shared.astype(float) ** 2 / (n_ref * n_cand), 0.0)
    score = merit if by == "merit" else np.where(is_host, shared, 0)
    order = np.argsort(score)[::-1]
    i, j = order[0], order[1]
    diag = dict(n_ref=n_ref, n_cand=int(cat.n_host[i]), shared=int(shared[i]),
                frac_recovered=shared[i] / n_ref, purity=shared[i] / n_cand[i],
                merit=merit[i], merit_2nd=merit[j], idx_2nd=int(j))
    diag["idx_best"] = int(i)
    diag["dominance"] = score[i] / max(score[j], 1e-30)
    ok = shared[i] >= min_shared and diag["dominance"] >= min_dominance
    return (int(i) if ok else -1), diag


def match_halo(halo, models, snaps, ref_set, ref_model="CDM", velo="velo"):
    """Match one zoom across models/snaps. `ref_set` = z=0 reference particle IDs
    (e.g. new_base_sets.pkl[halo]). Returns a list of row dicts."""
    stem = lambda m, s: f"{RUNS}/{m}/halo_{halo}/{velo}/veloshot_{s:04d}"
    ref = np.fromiter(ref_set, np.int64)
    rows = []
    # trace backwards one snapshot at a time: the z=0 core alone can share
    # almost nothing with a progenitor caught mid-merger
    for s in sorted(snaps, reverse=True):
        z_cdm = snapshot_redshifts(ref_model, halo).get(s)
        cdm = Catalogue(stem(ref_model, s))
        # 1) main progenitor of the previous CDM match (most shared particles)
        ci, cdiag = best_match(cdm, ref, by="shared", min_dominance=1.0)
        if ci >= 0 and cdiag["frac_recovered"] < 0.05:
            ci = -1
        cdm_ids = cdm.host_ids(ci) if ci >= 0 else None
        if cdm_ids is not None:
            ref = cdm_ids
        rows.append(_row(halo, ref_model, s, ci, cdiag, cdm, ci, cdm, z_cdm, s))
        for m in models:
            # SIDM output lists can differ: pick the SIDM snapshot at the same z
            zs = snapshot_redshifts(m, halo)
            ms = [k for k, z in zs.items() if z_cdm is not None and abs(z - z_cdm) < 0.005]
            if cdm_ids is None or not ms or not os.path.exists(stem(m, ms[0]) + ".properties"):
                rows.append(dict(halo=halo, model=m, snap=s, idx=-1, z=z_cdm,
                                 status="no_cdm_match" if cdm_ids is None else "no_snapshot_at_z"))
                continue
            cat = Catalogue(stem(m, ms[0]))
            # 2) same-time cross-model match to the CDM halo at this redshift
            mi, mdiag = best_match(cat, cdm_ids, by="merit")
            rows.append(_row(halo, m, s, mi, mdiag, cat, ci, cdm, z_cdm, ms[0]))
    return rows


_ZCACHE = {}


def snapshot_redshifts(model, halo):
    """{snapshot number: redshift} from the snapshot headers of one run."""
    key = (model, halo)
    if key not in _ZCACHE:
        d = f"{RUNS}/{model}/halo_{halo}/snap"
        out = {}
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.startswith("snapshot_") and f.endswith(".hdf5"):
                try:
                    with h5py.File(os.path.join(d, f), "r") as h:
                        out[int(f[9:13])] = float(h["Header"].attrs["Redshift"][0])
                except OSError as e:  # e.g. truncated snapshot
                    print(f"WARNING unreadable {d}/{f}: {e}", flush=True)
        _ZCACHE[key] = out
    return _ZCACHE[key]


def _row(halo, model, snap, idx, diag, cat, cdm_idx, cdm, z, model_snap):
    r = dict(halo=halo, model=model, snap=snap, z=z, model_snap=model_snap, idx=idx,
             status="ok" if idx >= 0 else "rejected", **diag)
    k = diag["idx_best"]
    if cdm_idx >= 0:
        pos = np.array([cat.props[c][k] for c in ("Xcminpot", "Ycminpot", "Zcminpot")])
        cpos = np.array([cdm.props[c][cdm_idx] for c in ("Xcminpot", "Ycminpot", "Zcminpot")])
        r.update(M200c=cat.props["Mass_200crit"][k],
                 offset_kpc=1e3 * np.linalg.norm(pos - cpos),
                 dlogM200c=np.log10(cat.props["Mass_200crit"][k] / cdm.props["Mass_200crit"][cdm_idx]))
    return r
