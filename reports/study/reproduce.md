# Reproduction

Run these commands from the repository root. Use Python 3.13 in Linux or WSL for NVIDIA GPU evaluation. The project requirements also support a CPU run with a compatible JAX installation. Rendering the included trajectories uses the recorded positions and events directly.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# NVIDIA GPU on Linux or WSL
python -m pip install 'jax[cuda12]==0.12.0.1'
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -c 'import jax; print(jax.devices())'
```

## Compute environment

The four-policy ownership study ran on an NVIDIA A100 with Python 3.12 and CUDA 12.6. Local evaluation and replay verification used an NVIDIA RTX 3080 Ti Laptop GPU. [Pinned dependencies](../../requirements.txt) and [GPU package versions](../../experiments/runtime-requirements.txt)

## Replay the recorded episodes

```bash
python src/render_replay.py reports/study/data/replays/environment --scenario-id 2 --output runs/review-environment.mp4 --stride 4
python src/render_replay.py reports/study/data/replays/dsac_gru --scenario-id 1000002 --output runs/review-dsac-gru.mp4 --stride 3
python src/render_replay.py reports/study/data/replays/ppo_mlp --scenario-id 1000002 --output runs/review-ppo-mlp.mp4 --stride 3
python src/render_replay.py reports/study/data/replays/dsac_gru_debt_1000 --scenario-id 3000063 --output runs/review-s28.mp4 --stride 4
```

## Evaluate the saved policy

Extract the archived simulator, trainer and evaluator from the completed ownership run. The archive uses the report templates from this repository. The [checkpoint manifest](checkpoints/manifest.json) records the public paths and SHA-256 hashes of the four checkpoints.

```bash
python -m zipfile -e reports/study/checkpoints/frozen_source.zip runs/study-frozen
python runs/study-frozen/src/evaluate.py \
  --checkpoint reports/study/checkpoints/dsac_gru.npz \
  --scenario-start 1000001 --episodes 64 --output runs/study-reproduction
```

The archived sampled evaluation and a local rerun both had 60/64 ownership completions. All 13 recorded endpoint arrays matched exactly. Add `--deterministic` and use a different output directory to compare with the archived deterministic audit of 59/64. Compare episode-level outcomes and inference mode before aggregating. Scenario IDs 1000001–1000064 are development IDs and include IDs used for checkpoint selection. [Reproduction receipt](data/diagnostics/reproduction.json), [frozen source](checkpoints/frozen_source.zip) and [archived evaluation](data/cohort/dsac_gru/evaluation.json)

## Run the 30-condition DSAC GRU evaluation

Extract the frozen archive as above. Run the same registered cases with the DSAC GRU checkpoint on an NVIDIA GPU:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false python experiments/evaluate_frozen_dsac_gru.py --source runs/study-frozen --output artifacts/dsac_gru_final
```

The runner verifies the archived source and selected policy checksums. It saves and audits every condition, and resumes from complete case artifacts when run again against the same directory. To extract the matched result tables and reconcile the 11 $1,000 winners from the raw logs, run `python reports/analyze_final_dsac_gru.py`. [Final evaluation record](final_dsac_gru.md)

## Retrain the original cohort

With the frozen source already extracted and the GPU environment active, run:

```bash
python runs/study-frozen/experiments/resource_study.py
```

The runner starts all four learners and writes into the extracted tree. It advances each learner through the recorded gates; the final evaluation stage depends on the cohort passing readiness. Training outputs need not be bitwise identical across GPU implementations. The archive includes the completed cohort’s source and settings. [Run history](experiments.md)

## Verify and regenerate figures

```bash
python src/verify.py
python src/analytic_checks.py
python src/verify_strategy.py
python src/verify_learning.py
python reports/build_study_figures.py
```

The figure script reads the included compact evidence. For diagnostics that use additional raw logs, set `GAMBLERS_TAXI_RECORDED_RUNS` to the directory containing the recorded runs. [Completed runner settings](data/cohort/freeze.json), [training protocol](training.md) and [experimental scope](experiments.md)
