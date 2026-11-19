#!/bin/bash
# Overnight (local, MPS/CPU only - no cluster jobs): probabilistic CDM -> SIDM converters on BAHAMAS DM-only maps,
# BAHAMAS -> BAHAMAS. 1) noise generators (gauss; displacement x1, x2, x3), 19 draws per test pair;
# 2) flow A on pairs_dm.npz (120 epochs, K=16, width 128), 19 draws; 3) comparison figures.
# Logs: runs_dm_overnight.log (this script), runs_dm_<step>.log
cd "$(dirname "$0")"
P=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
date
for m in "gauss" "displace" "displace --uscale 2" "displace --uscale 3"; do
  tag=$(echo $m | tr -d ' -' | sed 's/uscale/_u/')
  $P dm_generators.py --method $m --ndraw 19 > runs_dm_gen_$tag.log 2>&1; echo "generator $m exit $?"; date
done
$P dm_compare.py gauss displace displace_u2 displace_u3 > runs_dm_compare_noflow.log 2>&1; echo "compare (no flow) exit $?"; date
$P train_flow.py --model A --name flowA_dm --data pairs_dm.npz --epochs 120 --K 16 --hidden 128 > runs_dm_flowA.log 2>&1; echo "flowA_dm exit $?"; date
$P dm_flow_draws.py --name flowA_dm --ndraw 19 > runs_dm_flow_draws.log 2>&1; echo "flow draws exit $?"; date
$P dm_compare.py gauss displace displace_u2 displace_u3 flow > runs_dm_compare.log 2>&1; echo "compare exit $?"; date
