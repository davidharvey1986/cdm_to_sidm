#!/bin/bash
# Stars flow (step 2 of the DM -> stars -> gas chain): p(SIDM stars - CDM stars | CDM stars, CDM DM, Delta DM, sigma, z)
# trained with the TRUE SIDM dark matter; at test time Delta DM comes from flowA_dm_hyb draws. Same architecture as the DM flows.
cd "$(dirname "$0")"; PY=/Users/davidharvey/Work/cdm_to_sidm/bin/python
date; $PY prepare_stars_pairs.py
$PY train_flow.py --model A --name flowA_stars --data pairs_stars.npz --epochs 120 --K 16 --hidden 128 --noise 0.005 > runs_stars_train.log 2>&1
echo "train exit $?"; date
