"""Install remade images into the production image trees, keeping backups.

For each remade file <new>/<model>/<snap>/<halo>_<axis>.pkl:
  1. copy the existing <dest>/<model>/<snap>/<halo>_<axis>.pkl to
     <dest>_mismatched_backup/<model>/<snap>/  (once; never overwritten)
  2. copy the remade file over it (via .tmp + os.replace)
Files whose destination already carries the remake's 'matched_by' meta are
skipped, so the script can be rerun (e.g. when the last images finish).
The remade trees themselves are left in place.

usage (from create_obs/): python3 match_fix/install_remake.py {20kpc,1kpc} [--dry-run]
       elsewhere:          python3 install_remake.py <new_root> <dest_root> [--dry-run]
"""
import os, sys, shutil, pickle

TREES = {"20kpc": ("images_rematch", "images"), "1kpc": ("images_1kpc_rematch", "images_1kpc")}
args = [a for a in sys.argv[1:] if a != "--dry-run"]
dry = "--dry-run" in sys.argv
prod = args[0] if args[0] in TREES else "custom"
new_root, dest_root = TREES[prod] if prod in TREES else (args[0], args[1].rstrip("/"))
backup_root = dest_root + "_mismatched_backup"

n = dict(installed=0, skipped=0, backed_up=0)
for dirpath, _, files in os.walk(new_root):
    for f in sorted(files):
        if not f.endswith(".pkl"):
            continue
        rel = os.path.relpath(os.path.join(dirpath, f), new_root)
        src, dest, bak = (os.path.join(r, rel) for r in (new_root, dest_root, backup_root))
        if os.path.isfile(dest) and "matched_by" in pickle.load(open(dest, "rb"))[1]:
            n["skipped"] += 1
            continue
        if dry:
            n["installed"] += 1
            continue
        if os.path.isfile(dest) and not os.path.isfile(bak):
            os.makedirs(os.path.dirname(bak), exist_ok=True)
            shutil.copy2(dest, bak)
            n["backed_up"] += 1
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest + ".tmp")
        os.replace(dest + ".tmp", dest)
        n["installed"] += 1
print(prod, "(dry run)" if dry else "", n, f"backups in {backup_root}/")
