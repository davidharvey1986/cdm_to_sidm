#!/bin/bash
# Stars flow with stellar-scale augmentation (each pair's CDM and SIDM stars x 10^U(-0.5, 0.5), floor re-applied), so the
# flow cannot learn regression to the BAHAMAS stellar-to-halo relation (stars_scale_check.py). Otherwise as run_stars.sh.
# Waits for the gas flow evaluation to finish (GPU), then trains, draws (oracle + chain) and re-runs the checks.
cd "$(dirname "$0")"; PY=/Users/davidharvey/Work/cdm_to_sidm/bin/python
until grep -q "wrote figures/gas_compare" "$1" 2>/dev/null; do sleep 30; done; date
$PY train_flow.py --model A --name flowA_stars_aug --data pairs_stars.npz --epochs 120 --K 16 --hidden 128 --noise 0.005 --scale-aug 0.5 > runs_stars_aug_train.log 2>&1
echo "train exit $?"; date; grep "best val" runs_stars_aug_train.log
for m in true hyb; do $PY stars_flow_draws.py --name flowA_stars_aug --dm $m --tag _aug 2>&1 | grep -E "epoch|wrote|rror"; done
$PY stars_scale_check.py --flow --model=flowA_stars_aug 2>&1 | grep -vi warn | grep -vE "^f="
echo done; date
