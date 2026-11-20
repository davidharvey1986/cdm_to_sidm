#!/bin/bash
# Gas flow (step 3 of the DM -> stars -> gas chain):
# p(SIDM gas - CDM gas | CDM gas, CDM DM, CDM stars, dDM, dstars, sigma, z), trained with the TRUE SIDM DM and stars;
# at test time dDM and dstars come from the DM-flow and stars-flow draws. Same architecture as the DM and stars flows.
cd "$(dirname "$0")"; PY=/Users/davidharvey/Work/cdm_to_sidm/bin/python
date
$PY train_flow.py --model A --name flowA_gas --data pairs_gas.npz --epochs 120 --K 16 --hidden 128 > runs_gas_train.log 2>&1
echo "train exit $?"; date
