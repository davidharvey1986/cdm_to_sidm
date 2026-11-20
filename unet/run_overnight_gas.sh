#!/bin/bash
# Overnight, after run_stars_aug.sh (stars flow with stellar-scale augmentation):
# 1) decision: PASS if the augmented stars flow is close to scale-free (|d log M*(<60 kpc)| shift between input x0.5 / x2
#    and x1 below 0.04 dex at every sigma; baseline 0.07-0.11) and its best val NLL is within 0.05 of flowA_stars (-1.8517);
# 2) gas scale check of the current gas flow (flowA_gas), either way;
# 3) GAS HYBRID flowA_gas_hyb: converter mean (prepare_gas_hybrid.py, data/gas_hyb_offset.npz) + flow on the residual;
#    with the scale augmentation if PASS (gas x 10^U(-0.5,0.5) hard floor, CDM stars channel its own factor), chain on
#    stars_hyb_aug if PASS else stars_hyb; draws (oracle + chain), scale check, gas_compare -> figures/gas_compare_stats_hyb.png;
# 4) if PASS: the non-hybrid augmented gas flow flowA_gas_aug as well (draws, scale check, gas_compare _aug).
cd "$(dirname "$0")"; PY=/Users/davidharvey/Work/cdm_to_sidm/bin/python
until grep -q "^done" runs_stars_aug.log 2>/dev/null; do sleep 60; done; date
$PY - <<'PY' > runs_overnight_decision.txt
import re
L = open("runs_stars_aug.log").read(); v = {}
for s, f, d in re.findall(r"flow SIDM(\S+) stars x(\S+): median d log M\*\(<60kpc\) (\S+)", L): v[s, f] = float(d)
try:
    bv = float(re.search(r"best val (\S+)", L).group(1))
    sh = max(abs(v[s, f] - v[s, "1"]) for s in ("0.1", "0.3", "1") for f in ("0.5", "2"))
    ok = sh < 0.04 and bv < -1.8517 + 0.05; print(f"max scale shift {sh:.3f} dex, best val {bv:.4f} -> {'PASS' if ok else 'FAIL'}")
except Exception as e: print(f"could not evaluate ({e!r}) -> FAIL")
PY
cat runs_overnight_decision.txt
if grep -q PASS runs_overnight_decision.txt; then AUG="--scale-aug 0.5 --floor-mode hard --scale-aug-cx 1"; ST=stars_hyb_aug; else AUG=""; ST=stars_hyb; fi
echo "gas hybrid augmentation: '${AUG}', chain stars: ${ST}"
$PY gas_scale_check.py --flow --model=flowA_gas 2>&1 | grep -vi warn | grep -E "SIDM|wrote|rror"
date; $PY train_flow.py --model A --name flowA_gas_hyb --data pairs_gas.npz --offset gas_hyb_offset.npz --epochs 120 --K 16 --hidden 128 ${AUG} > runs_gas_hyb_train.log 2>&1
echo "gas hybrid train exit $?"; date; grep "best val" runs_gas_hyb_train.log
for m in true hyb; do $PY gas_flow_draws.py --name flowA_gas_hyb --dm $m --tag _hyb --stars ${ST} 2>&1 | grep -E "epoch|wrote|rror"; done
$PY gas_scale_check.py --flow --model=flowA_gas_hyb 2>&1 | grep -vi warn | grep -E "SIDM|wrote|rror"
$PY gas_compare.py "oracle,chain,converter,hybrid oracle,hybrid chain" _hyb 2>&1 | grep -vi warn | grep -E "SIDM|wrote|rror"
if grep -q PASS runs_overnight_decision.txt; then
  date; $PY train_flow.py --model A --name flowA_gas_aug --data pairs_gas.npz --epochs 120 --K 16 --hidden 128 ${AUG} > runs_gas_aug_train.log 2>&1
  echo "gas aug train exit $?"; date; grep "best val" runs_gas_aug_train.log
  for m in true hyb; do $PY gas_flow_draws.py --name flowA_gas_aug --dm $m --tag _aug --stars stars_hyb_aug 2>&1 | grep -E "epoch|wrote|rror"; done
  $PY gas_scale_check.py --flow --model=flowA_gas_aug 2>&1 | grep -vi warn | grep -E "SIDM|wrote|rror"
  $PY gas_compare.py "oracle,chain,hybrid chain,aug oracle,aug chain" _aug 2>&1 | grep -vi warn | grep -E "SIDM|wrote|rror"
fi
echo done; date
