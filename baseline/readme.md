# Gambler’s Taxi

A JAX driving simulator for studying long-horizon decisions under risk. Drive through traffic, work, rest, gamble, and manage debt. In the main campaign, clear $1,000 at the bank and buy out the taxi before the ten-day deadline.

[Environment report](reports/assignment1.md), [Reinforcement learning report](reports/assignment2.md), [research report](reports/debt_research.md) and [Learned $1,000 buyout video](artifacts/debt_research/cases/seed_91038.mp4)

## What is included

- A `BaseEnv`-derived vehicle environment with lidar, bicycle dynamics, static and moving obstacles, and a playable Pygame frontend.
- An end-to-end Hybrid DSAC agent with continuous driving controls, discrete operations, and quantile critics.
- A destination PPO study with a shared driver, scripted controls, and tests that distinguish voluntary repeat bets from environment overrides.
- Saved weights, per-episode results, intervention data, and numeric traces. Video rendering runs after training and simulation.

On 128 held-out starts per policy, work-trained destination agents bought out in **75–104** episodes. Mixed work-and-casino agents bought out in **5–29**. A driving guard substantially reduced crashes for the archived Hybrid DSAC actors, while both still failed to reach a legal repayment. [Full results and limitations](reports/debt_research.md)

## Quick start

Obtain this project repository using Git or download and extract its source archive. Run the commands from the directory containing `requirements.txt`. Python 3.13 is the reference version. CPU is sufficient for play, transition checks, and rendering saved logs.

**Linux or WSL2**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/verify.py
python src/play.py --dependence
```

**Windows PowerShell**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/verify.py
.\.venv\Scripts\python.exe src/play.py --dependence
```

Use W/S or ↑/↓ to accelerate and brake, A/D or ←/→ to steer, 1–0 for the operations shown in the panel, R to reset, and Esc to exit. `--dependence` enables the optional habit and impulse mechanics in manual play.

**NVIDIA GPU training on Linux or WSL2**

```bash
python -m pip install "jax[cuda12]==0.12.0.1" "numpy==2.5.4"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
python -c "import jax; print(jax.devices())"
```

The reference software and hardware are Ubuntu WSL2, Python 3.13, and an RTX 3080 Ti Laptop GPU. The reproduction check used Python 3.13.13. The device check should list a CUDA device. The [JAX install guide](https://docs.jax.dev/en/latest/installation.html) covers driver and platform requirements.

## Replay the results

Render saved trajectories on CPU or GPU. On Windows, use the virtual environment's Python executable shown above.

```bash
python src/replay.py render --log artifacts/debt_work_episode.npz --video runs/replay/assignment1.mp4
python src/replay.py render --log artifacts/policy_episode.npz --video runs/replay/assignment2.mp4
python src/replay.py render --log artifacts/debt_research/cases/seed_91038_render.npz --video runs/replay/risky-buyout.mp4
```

To run new inference from a saved end-to-end model:

```bash
python src/replay.py record --checkpoint artifacts/early_neutral.npz --seed 20000 --log runs/replay/new-hybrid-episode.npz
```

To re-run the destination policy behind the risky buyout on the original test batch:

```bash
python src/debt_replay.py --research artifacts/debt_research --policy mixed_demos_urgency_1_seed_29 --seed 91038 --log runs/replay/new-risky-buyout.npz
```

That command checks outcome, terminal time, cash, and debt against the original evaluation. Current and historical trajectories are preserved in the saved logs. Fresh inference can vary near collision boundaries across devices and compiled loops.

## Reproduce the research

Use the GPU environment above. Each new run writes under `runs/`. The reference data in `artifacts/` can then be compared against the regenerated outputs.

### Evaluate the saved policies

Copy the two archived Hybrid DSAC checkpoints next to the output directory, and place the PPO models inside it. The evaluator uses these relative locations.

```bash
mkdir -p runs/debt-checkpoints
cp artifacts/debt_neutral_actor.npz artifacts/debt_upside_actor.npz runs/
cp -r artifacts/debt_research/models runs/debt-checkpoints/
python src/debt_research.py --output runs/debt-checkpoints
python src/debt_decisions.py --output runs/debt-checkpoints
```

`results.json` records configuration, seed banks, file hashes, outcome counts, and summaries. `*_episodes.json` contains individual episodes, `*_probes.npz` contains paired actor outputs, and `decision_probes.json` contains the work, home, and bank interventions. The saved evaluation contains 18 conditions. The verification rerun matched the 14 destination-condition summaries. Exact crash and wager counts for the old continuous actors varied, while the zero-buyout result and the crash reduction from guarding persisted. [Reference results](artifacts/debt_research/results.json)

### Train the nine destination policies

```bash
mkdir -p runs/debt-retrained
python src/debt_learning.py --output runs/debt-retrained/models --iterations 50
python src/debt_learning.py --output runs/debt-retrained/models --iterations 50 --urgency 1 --expert mixed
python src/debt_research.py --output runs/debt-retrained --destination-only
python src/debt_decisions.py --output runs/debt-retrained
```

The first command trains six policies across two debt-cost values and three seeds. The second trains three mixed-demonstration policies. Each saves `imitation.npz`, `selected.npz`, `last.npz`, and `validation.json`. The current trainer records imitation weights for all nine runs. The archived evaluation includes imitation-only policies from the three mixed runs.

For the same-seed repeat, run the command below and compare the weight arrays `w0`–`w2` and `b0`–`b2` from `selected.npz` and `last.npz`, plus `validation.json`. Compare arrays rather than archive bytes because metadata can change. [Saved repeat check](artifacts/debt_research/repeat_check.json)

```bash
python src/debt_learning.py --output runs/debt-repeat --iterations 50 --urgency 0.25 --seeds 29
```

### Train the end-to-end debt agents

The commands reuse the saved eight-day actor and run imitation, corrective demonstrations, and Hybrid DSAC with the current ten-day environment. The neutral reference uses seed 11. The upper-tail series uses seeds 11, 29, and 47. Each run writes its own `best.npz`, `last.npz`, metrics, and training history and evaluates the original test bank (20000–20127).

```bash
python src/train.py --output runs/hybrid-neutral-11 --seed 11 \
  --initialize artifacts/hard_world_actor.npz --consistent-expert \
  --iterations 650 --imitation 6000 --correction-rounds 4 --correction-steps 1600 \
  --demo-weight 7 --learning-rate 0.00010 --updates 8 \
  --validate-every 25 --log-every 50 --upside 0

for seed in 11 29 47; do
  python src/train.py --output "runs/hybrid-upside-$seed" --seed "$seed" \
    --initialize artifacts/hard_world_actor.npz --consistent-expert \
    --iterations 650 --imitation 6000 --correction-rounds 4 --correction-steps 1600 \
    --demo-weight 7 --learning-rate 0.00010 --updates 8 \
    --validate-every 25 --log-every 50 --upside 0.55
done
```

Earlier scenarios used different environment and demonstration code. The archived checkpoints, metrics, and logs reproduce their recorded results and videos. Training those historical configurations from scratch requires the corresponding source revision.

### Benchmarks and figures

```bash
python src/benchmark.py
python -m pip install matplotlib
python reports/plot_research.py
```

The figure script reads the saved evaluation and trajectory files and regenerates the figures in `reports/figures/`.

## Project layout

| Path | Contents |
|---|---|
| [src/env/](src/env/) | Vehicle, city, economy, observations, and rewards |
| [src/algorithms/](src/algorithms/) | Hybrid DSAC |
| [src/train.py](src/train.py) | End-to-end training and evaluation |
| [src/debt_learning.py](src/debt_learning.py) | Destination PPO |
| [src/debt_research.py](src/debt_research.py) | Held-out evaluation and paired gambling probes |
| [src/debt_decisions.py](src/debt_decisions.py) | Controlled input interventions |
| [reports/](reports/) | Two IAEML reports and the report |
| [artifacts/](artifacts/) | Checkpoints, metrics, logs, and videos |

The environment and renderer build on the [InnoRL IAEML starter repository](https://github.com/InnoRL/IAEML-assignment). The reports cite the algorithm and behavioral research used in the project.
