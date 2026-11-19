"""Check remade images (run from create_obs/ with the system python).

For every row of the remake table and axis: the file exists, has the expected
shape and pixel size, is centred on the particle-matched VR halo
(meta vr_index == idx_best, props ID == idx_best + 1) and is at the CDM
redshift. Also compares the central mass (within R_AP kpc of the map centre)
with the CDM image of the same halo, for both the new and the old SIDM image.

usage: python3 match_fix/verify_remake.py {20kpc,1kpc}
"""
import os, sys, csv, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {"20kpc": dict(new="images_rematch", old="images", table="images_to_remake.csv",
                     shape=(1, 100, 100), pix=20.0),
       "1kpc": dict(new="images_1kpc_rematch", old="images_1kpc", table="images_1kpc_to_remake.csv",
                    shape=(2, 1000, 1000), pix=1.0)}
R_AP = 200.0  # kpc


def load(path):
    maps, meta = pickle.load(open(path, "rb"))
    return np.asarray(maps), meta


def central_mass(maps, pix):
    m = maps[0]
    n = m.shape[-1]
    y, x = np.indices(m.shape)
    r = np.hypot(x - (n - 1) / 2, y - (n - 1) / 2) * pix
    return m[r <= R_AP].sum() * (pix / 1e3) ** 2  # Msun (map in Msun/pMpc^2)


prod = sys.argv[1]
c = CFG[prod]
rows = list(csv.DictReader(open(os.path.join(HERE, c["table"]))))
out, problems = [], []
for r in rows:
    snap, model, halo, idx = int(float(r["snap"])), r["model"], r["halo"], int(float(r["idx_best"]))
    for ax in "xyz":
        f = f"{c['new']}/{model}/{snap}/{halo}_{ax}.pkl"
        if not os.path.isfile(f):
            problems.append(f"missing {f}")
            continue
        maps, meta = load(f)
        cdm_maps, cdm_meta = load(f"{c['old']}/CDM/{snap}/{halo}_{ax}.pkl")
        old_maps, _ = load(f"{c['old']}/{model}/{snap}/{halo}_{ax}.pkl")
        bad = []
        if maps.shape != c["shape"]:
            bad.append(f"shape {maps.shape}")
        if abs(meta["pixel_size_pkpc"] - c["pix"]) > 1e-6:
            bad.append(f"pix {meta['pixel_size_pkpc']}")
        if meta["vr_index"] != idx or meta["props"]["ID"] != idx + 1:
            bad.append(f"idx {meta['vr_index']} ID {meta['props']['ID']} want {idx}")
        if abs(meta["redshift"] - cdm_meta["redshift"]) > 0.005:
            bad.append(f"z {meta['redshift']} vs CDM {cdm_meta['redshift']}")
        if not np.isfinite(maps).all():
            bad.append("non-finite")
        if bad:
            problems.append(f"{f}: " + "; ".join(bad))
        mc = central_mass(cdm_maps, c["pix"])
        out.append(dict(model=model, snap=snap, halo=halo, axis=ax,
                        ratio_new=central_mass(maps, c["pix"]) / mc,
                        ratio_old=central_mass(old_maps, c["pix"]) / mc))

print(f"{prod}: {len(out)} images checked, {len(problems)} problems")
for p in problems:
    print("  ", p)
rn = np.array([o["ratio_new"] for o in out])
ro = np.array([o["ratio_old"] for o in out])
q = lambda a: " ".join(f"{v:.2f}" for v in np.percentile(a, [5, 25, 50, 75, 95]))
print(f"M(<{R_AP:.0f} kpc) SIDM/CDM, 5/25/50/75/95 %:")
print("   new:", q(rn))
print("   old:", q(ro))
print("   |log ratio| < 0.15 (within ~40%):  new", (np.abs(np.log10(rn)) < 0.15).sum(),
      " old", (np.abs(np.log10(ro)) < 0.15).sum(), "of", len(out))
worst = sorted(out, key=lambda o: -abs(np.log10(o["ratio_new"])))[:6]
print("largest new deviations:")
for o in worst:
    print(f"   {o['model']} {o['snap']} {o['halo']} {o['axis']}  new {o['ratio_new']:.2f}  old {o['ratio_old']:.2f}")
with open(os.path.join(HERE, f"verify_{prod}.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(out[0]))
    w.writeheader()
    w.writerows(out)
