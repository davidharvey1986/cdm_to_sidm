# cdm_to_sidm

Turn a suite of cold-dark-matter (CDM) cluster simulations into self-interacting-dark-matter (SIDM) versions.
Trained and tested on BAHAMAS, mapping CDM to SIDM at cross-sections 0.1, 0.3 and 1 cm²/g. Training is within one
suite at a time; suites are never combined.

## Layout
- `unet/`: the converters and their checks.
  - `two_scale.py`: two-scale mean converter: a radial core template plus a mapped small-scale term.
  - `flows.py`, `train_flow.py`: conditional normalising flow p(SIDM − CDM | CDM map, σ, z).
  - Dark-matter-only probabilistic converter:
    - `prepare_dm_pairs.py`: builds the dark-matter-only pairs;
    - `prepare_dm_hybrid.py`: hybrid targets, converter mean plus flow on the residual;
    - `dm_flow_draws.py`: draws from a trained flow;
    - `dm_compare.py`: compares the draws with the true SIDM maps.
- `velmaps/`: dark-matter line-of-sight velocity-moment maps.
  - `velmaps/bahamas/bahamas_vel_maps.py`: BAHAMAS velocity maps, built from particles on COSMA.
  - `velmaps/bahamas/bahamas_baryon_maps.py`: gas and stellar mass maps on the same grid.
- `bahamas/`: builds the matched BAHAMAS CDM/SIDM stamp database.
- Top level: halo matching and image remaking for the darkskies zooms.

Data, catalogues, trained models and figures are not in the repository.
