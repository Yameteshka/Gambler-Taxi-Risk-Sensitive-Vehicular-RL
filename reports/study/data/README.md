# Data files

| Location | Contents |
|---|---|
| [Development summary](development_summary.csv) | One row per selected policy with outcomes, passed lessons and both experience budgets |
| [Selected cohort](cohort/) | Four evaluation summaries, endpoint arrays, training and gate logs, configs, and completion or failure receipts |
| [Final DSAC GRU data](final_dsac_gru/) | All 30 conditions, endpoint arrays, audits, paired outcomes, behavior counts and $1,000 transaction ledgers |
| [Final condition register](frozen_conditions.csv) | S01–S30, planned policy and episode counts, and recorded completion status |
| [Diagnostics](diagnostics/) | Retention, critic outputs, actor gradients, navigation, menus, and environment checks |
| [Recovery trials](recovery_trials/) and [smoke tests](smoke/) | Local continuations, observation tests, PPO stress runs and policy replay audit |
| [Trace index](trace_index.json) and [replays](replays/) | Development paths and training fragments for plots. Replay folders include all available fields for the individual video episodes. |
| [File index](local_file_index.csv) | Published paths and file sizes |
| [Evaluation records](local_evaluations.csv) and [historical videos](historical_videos.csv) | Evaluation metrics and rendered clips |
| [Inventory summary](inventory_summary.json) | Counts and scope of the file inventory |

Inventory paths are relative to the repository root. Evaluation records whose raw files are unavailable use record IDs. [Experimental denominators](../results.md)
