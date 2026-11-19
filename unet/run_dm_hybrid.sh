#!/bin/bash
# Hybrid DM converter: two-scale converter mean + flow on the residual (data/pairs_dm_hyb.npz), then draws and comparison.
cd "$(dirname "$0")"; PY=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
date; $PY train_flow.py --model A --name flowA_dm_hyb --data pairs_dm_hyb.npz --epochs 120 --K 16 --hidden 128 > runs_dm_hyb_train.log 2>&1; echo "train exit $?"
date; $PY dm_flow_draws.py --name flowA_dm_hyb --ndraw 19 --hybrid > runs_dm_hyb_draws.log 2>&1; echo "draws exit $?"
date; $PY dm_compare.py gauss displace_u3 flow flow_hyb > runs_dm_hyb_compare.log 2>&1; echo "compare exit $?"; date
