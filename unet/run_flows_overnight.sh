#!/bin/bash
# Overnight: train flow A and flow B (120 epochs each, K=16, width 128), then evaluate on the test split.
# Local (MPS) only - no cluster jobs. Logs: runs_flowA.log, runs_flowB.log, runs_flow_eval.log
cd "$(dirname "$0")"
P=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
date
$P train_flow.py --model A --name flowA --epochs 120 --K 16 --hidden 128 > runs_flowA.log 2>&1
echo "flowA exit $?"; date
$P train_flow.py --model B --name flowB --epochs 120 --K 16 --hidden 128 > runs_flowB.log 2>&1
echo "flowB exit $?"; date
$P evaluate_flows.py --A flowA --B flowB --split test --n 16 > runs_flow_eval.log 2>&1
echo "eval exit $?"; date
