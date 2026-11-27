# Figures and video

## Figures

| Figure | What it shows |
|---|---|
| [Environment](../../media/figures/environment_model.svg) | Vehicle controls, services and terminal conditions |
| [Model architectures](../../media/figures/agent_architectures.svg) | DSAC and PPO decision and update loops. Marker strokes identify how their GRUs use history and where each learner gets its update data. |
| [Training and evaluation](../../media/figures/training_pipeline.svg) | Simulation, data collection, development gates and saved outputs |
| [Training curriculum](../../media/figures/ownership_curriculum.svg) | The progression from short reset contexts to the full campaign |
| [Development results](../../media/figures/development_results.svg) | Ownership on 64 scenarios and lessons passed by each policy |
| [Curriculum progress](../../media/figures/curriculum_progress.svg) | Gates passed at recorded network decision counts for each learner |
| [Final condition outcomes](../../media/figures/final_condition_outcomes.svg) | Terminal outcomes in all 30 DSAC GRU tests |
| [Matched development routes](../../media/figures/matched_paths.svg) | All four policies on the same 16 scenario IDs, with scenario 1000002 emphasized |
| [Training fragments and frozen routes](../../media/figures/training_and_transfer_paths.svg) | Logged DSAC GRU fragments from three lessons and routes from three frozen checkpoints |
| [Resource and debt traces](../../media/figures/resource_traces.svg) | The two replayed policies on scenario 1000002 |

Each translucent line comes from recorded positions. Terminal markers show where episodes ended. The training panels show fragments from early collection chunks, with up to 12 traces per lesson. The development panels use the first 16 common scenarios. The route plots above use the development bank. The final replay below comes from condition S28. [Trace index](data/trace_index.json) and [figure script](../build_study_figures.py)

## Videos

| Video | Episode | Evidence |
|---|---|---|
| [Environment demonstration](../../media/video/environment-demonstration.mp4) | Reference controller on scenario 2 with $1,000 car principal | [Replay input](data/replays/environment/) and [events](../../media/video/environment-demonstration.events.json) |
| [DSAC GRU ownership](../../media/video/dsac-gru-1000002.mp4) | Learned policy, $500 development scenario 1000002. Ownership at tick 571. | [Replay input](data/replays/dsac_gru/), [events](../../media/video/dsac-gru-1000002.events.json) and [loan ledger](../../media/video/dsac-gru-1000002.loans.json) |
| [PPO MLP starvation](../../media/video/ppo-mlp-1000002.mp4) | Learned policy, same development scenario. Starvation at tick 349. | [Replay input](data/replays/ppo_mlp/), [events](../../media/video/ppo-mlp-1000002.events.json) and [loan ledger](../../media/video/ppo-mlp-1000002.loans.json) |
| [DSAC GRU $1,000 buyout](../../media/video/dsac-gru-debt-1000-3000063.mp4) | Final condition S28, scenario 3000063. Five top-rate shifts, ownership at tick 1114. | [Replay input](data/replays/dsac_gru_debt_1000/), [events](../../media/video/dsac-gru-debt-1000-3000063.events.json), [transaction data](data/final_dsac_gru/s28_buyout_events.csv) |

The renderer creates English lines by looking up logged events and formatting their amounts and balances. Each line describes a recorded event. [Renderer](../../src/render_replay.py) and [historical media index](data/historical_videos.csv)
