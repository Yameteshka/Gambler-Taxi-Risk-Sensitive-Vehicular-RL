"""Build English reports exclusively from recorded results and registered status."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def inventory():
    with (ROOT/"experiments/registry.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def recorded():
    records = []
    for path in sorted((ROOT/"artifacts/evaluation").glob("*/*/evaluation.json")):
        if not (path.parent/"complete.json").exists():
            continue
        audit = path.parent/"audit.json"
        if not audit.exists() or not json.loads(audit.read_text(encoding="utf-8")).get("passed"):
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        item.update(run_id=path.parent.parent.name, case=path.parent.name)
        records.append(item)
    return records


def table(records):
    lines = ["| Run | Case | Episodes | Ownership | Crashes | Starvation | Debt |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['run_id']} | {r['case']} | {r['episodes']} | {r['ownership']} | {r['collision']} | {r['starvation']} | ${r['mean_remaining_debt']:.2f} |")
    return chr(10).join(lines) if records else "No final held-out evaluations have completed. "


def figures(rows, records):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False, False
    target = ROOT/"runs/legacy_report_figures"
    target.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    curves = 0
    for row in rows:
        if row["group"] != "main" or row["training_horizon"] != "10d" or row["repetition"] != "A":
            continue
        path = ROOT/"artifacts/training"/row["run_id"]/"validation.json"
        if path.exists() and (history := json.loads(path.read_text(encoding="utf-8"))):
            ax.plot([r["physical_transitions"] for r in history], [r["success_rate"] for r in history],
                    marker=".", label=f"{row['algorithm']} / {row['architecture']} / A")
            curves += 1
    ax.set(xlabel="Physical environment transitions", ylabel="Validation ownership rate", ylim=(-.02, 1.02))
    ax.set_title("Learning curves — declared initialization A")
    if curves:
        ax.legend()
        ax.grid(alpha=.2)
        fig.savefig(target/"learning_curves.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    colors = ("#2a9d8f", "#e76f51", "#d9a441", "#748cab")
    groups = []
    for algorithm, architecture in (("hybrid_dsac", "mlp"), ("hybrid_ppo", "mlp"), ("hybrid_ppo", "gru")):
        ids = {r["run_id"] for r in rows if r["group"] == "main" and r["training_horizon"] == "10d" and
               r["algorithm"] == algorithm and r["architecture"] == architecture}
        items = [r for r in records if r["run_id"] in ids and r["case"] == "standard"]
        if items:
            groups.append((f"{algorithm}\n{architecture} (n={len(items)} runs)",
                [np.mean([r[k]/r["episodes"] for r in items]) for k in ("ownership", "collision", "starvation", "deadline")]))
    if groups:
        bottom = np.zeros(len(groups))
        for i, label in enumerate(("Ownership", "Collision", "Starvation", "Deadline")):
            values = np.array([g[1][i] for g in groups])
            ax.bar(range(len(groups)), values, bottom=bottom, color=colors[i], label=label)
            bottom += values
        ax.set_xticks(range(len(groups)), [g[0] for g in groups])
        ax.legend(loc="upper center", ncol=4)
        ax.set(ylabel="Mean fraction of 256 held-out scenarios", ylim=(0, 1.15), title="Ten-day task outcomes")
        fig.savefig(target/"task_outcomes.png", dpi=180)
    plt.close(fig)
    return bool(groups), bool(curves)


COMMON = """
All external quotes and outcomes are generated from scenario-specific named random streams. Initialization seeds A–E (201–205) control learning randomness, separately from scenario IDs. Development IDs 1–10 are never used for training, reset curricula, demonstrations, or replay. Training uses 11–10010; validation uses 1000001–1000064; the standard final bank is 2000001–2000256. Adverse banks have separate non-overlapping ranges in `experiments/protocol.json`. Increasing the training scenario count does not move reserved banks; overlap is rejected before training.

Each trained policy is evaluated on the same 256 final scenarios. A fifth percentile uses approximately 13 tail observations. Each evaluation records Wilson intervals for ownership and a scenario bootstrap for mean return, p05 and lower-tail CVaR05. Training variability is assessed across five fresh initializations. Matched comparisons use crossed resampling over initialization and scenario; treating all 1280 outcomes as independent would understate uncertainty. 
"""


def main():
    rows, results = inventory(), recorded()
    target = ROOT/"reports"
    target.mkdir(exist_ok=True)
    from statistics_report import registered_comparisons
    registered_comparisons(ROOT)
    trained = sum((ROOT/"artifacts/training"/r["run_id"]/"complete.json").exists() for r in rows)
    evaluated = len({r["run_id"] for r in results})
    status = f"Recorded status: {trained}/{len(rows)} registered training runs completed; {evaluated} policies have final standard evaluations."
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    have_outcomes, have_curves = figures(rows, results)
    plot1 = "![Ten-day outcomes](../runs/legacy_report_figures/task_outcomes.png)" if have_outcomes else ""
    plot2 = "![Validation learning curves](../runs/legacy_report_figures/learning_curves.png)" if have_curves else ""
    analytic_path = ROOT/"experiments/analytic_checks.json"
    analytic = json.loads(analytic_path.read_text(encoding="utf-8")) if analytic_path.exists() else {}
    bj = analytic.get("blackjack_optimal_expected_net_per_dollar", float("nan"))
    a1 = f"""# Assignment 1: Gambler's Taxi Environment

Iuliia Kurnaeva  
{now}

## Scope and development

The initial idea was a choice between stable employment and uncertain casino payouts. The course requires a vehicle environment, so services became destinations on a road map. The current task studies financial planning under survival costs: the driver starts at home with $100, a $1000 car loan, fuel and food reserves. The goal is to repay both car debt and every personal loan, then reach the garage to complete ownership. Hunger, fuel, fatigue and travel time create competing uses of cash and time. Mood, anxiety, imposed gambling impulses, dismissal and the job center were removed. Work remains available at full fatigue; running out of fuel allows slow pushing, and running out of satiety causes death.

The environment is implemented in a new file, [resource_taxi.py](../src/env/resource_taxi.py), as a subclass of the course `BaseEnv`, with `init_params`, `reset`, `step` and `get_observation`. The original `base.py` is preserved. The map is 40 × 30, with ten service zones, static circular obstacles and three moving circular vehicles. Both the agent and obstacle circles have radius one; swept collisions use the sum of radii. This explicitly resolves an ambiguity in the starter collision helper, whose default center-distance threshold is only one. The course's lidar intersection implementation is reused. Static route fields are precomputed once for all services and supplied as distances and local directions. They account for static geometry, while moving traffic is sensed locally. They do not select the agent's economic destination.

## Dynamics, state and observations

A kinematic bicycle model offers a turning radius and braking distance at a low simulation cost. The continuous actions are acceleration/braking and target steering, each normalized to [-1,1]. A categorical operation branch chooses drive, wait, work, sleep, eat, two refueling quantities, three borrowing amounts, six repayment choices, three slot stakes, three blackjack stakes, hit, stand or ownership completion. There are 25 physical categorical choices. The learned service interface maps 26 intentions onto these physical operations and a shared route controller, with 179 public input features. Operations require parking within 1.55 map units at speed below 0.2. Eligibility depends on location, service state, cash, opening hours and debt; it does not ban work or gambling at full fatigue.

With dt=0.16, wheelbase=0.8 and steering limit=0.58 radians, steering approaches its target at a fatigue-dependent rate. Traction is 2.9 times positive acceleration; braking is 5.1 times the negative command, reduced by fatigue. Positive-speed drag is 0.22+0.055 v². Speed is clipped to [0,3.8]. Translation and yaw use the mean of old and new speed and a midpoint heading. This is a kinematic model, not a tire-slip model or a claim of realistic passenger-car dynamics.

```text
steer' = steer + (0.72 - 0.36 fatigue) × control_strength × (command × 0.58 - steer)
v' = clip(v + dt × (traction_or_braking - drag), 0, vmax)
yaw_increment = dt × mean(v,v') × tan(steer') / wheelbase
position' = position + dt × mean(v,v') × direction_at_midpoint
```

No fuel means voluntary low-speed pushing, capped at 0.2, with greater hunger and fatigue cost. The vehicle can be pushed anywhere, rather than being teleported to a station. Free Wait uses zero controls and coasts under friction; it does not stop momentum instantly. Timed services hold the parked car stationary while traffic, hunger and interest continue. Sleep lasts 24 ticks. A work shift lasts 28 ticks and may start during the first 70 ticks of each 155-tick day, once per day. It consumes fuel. Daily wages are $175 times 0.75, 1 or 1.35, with probabilities 0.3, 0.4 and 0.3. The provisional wage was $145; development-only calibration showed insufficient cash for the diagnostic strategy, so it was increased before final testing.

The numerical state includes position, heading, speed, steering, clock, service progress, cash, outstanding car principal and charges, satiety, fuel, fatigue, cards, daily shift status, counters and individual loan arrays. The public observation contains 16 lidar lengths, service-relative bearings and route gradients, resources, debts, rates, public quotes, clock and horizon, service status, previous public action/event, and blackjack information. Hidden cards and future random draws are excluded. The recurrent extensions may retain history. State and observations retain the base renderer field names such as `agent_pos`, `goal_pos`, `path_array` and `collision_rays`.

## Accounts, food and games

Cash has no $100 cap after reset. Car installments require cumulative principal repayment of 30%, 60% and 100% at days 3, 6 and 10; a missed checkpoint adds a $10 charge once. The five-day campaign uses the same calendar in the main comparison. Open-ended campaigns retain the contract calendar but have no game deadline; evaluation is administratively censored at 30 days. The accelerated five-day calendar is a separate extension.

Voluntary bank loans of $25/$50/$100 create distinct IDs and terms. The car rate is 0.5% daily and personal loans charge 3% daily, accrued per tick using (1+r)^(1/155)-1. Principal repayment and fees are accounted for separately. Repayment of personal loans follows highest rate, then oldest ID. New borrowing increases cash and liabilities equally. Personal debt prevents ownership even when the car balance is zero. Loan storage can grow during continuing training; evaluation storage is allocated from its observation window. These are implementation allocations, not a credit limit.

| Food service | Price draw per day | Portion draw per purchase | Service time |
|---|---|---|---:|
| Budget diner near fuel | U($4,$6) | U(0.04,0.06) | 2 ticks |
| Distant cafe | U($12,$18) | U(0.12,0.18) | 5 ticks |
| Distant restaurant | U($30,$40) | U(0.55,0.75) | 10 ticks |

Daily prices stay fixed within a day. Portions are sampled once per purchase and delivered after service, provided the driver survives. Satiety is clipped at one, so a large meal can be partly wasted. No venue guarantees full restoration. Cafe and restaurant sites have nearly equal static route distances from home and work; the verification tolerance is 5%. The cheap diner is deliberately closer. Dining does not directly remove fatigue. Deterministic-mean and higher-variance matched-mean food conditions are registered separately.

Slots have 12% win probability, gross payout multiplier 325/55 and stakes $10/$25/$55. Expected net return is -29.09% of stake. Blackjack uses replacement cards, soft aces, a dealer standing on 17, natural payouts and explicit hit/stand. A 10% commission applies to winning profit. Exact dynamic programming for this implemented model gives optimal expected net return {100*bj:.2f}% of stake. The independent-outcome core does not make a table due to win after losses. Opening hours, service time, shared cooldown and travel cost constrain gambling. Casino visits receive no special reward, and no action override forces a bet.

## Reward and safeguards

All principal algorithms share the external reward: +1 for ownership plus a cash reserve bonus capped at 0.05, -0.1 on any unsuccessful terminal, and -0.1(1-gamma) on each nonterminal step. Terminal steps have no additional running charge. For an unsuccessful episode of T physical steps, the discounted base return is exactly -0.1, independent of failure time. For ownership it is -0.1 + gamma^(T-1)(1.1 + reserve_bonus), so earlier otherwise identical success is strictly preferred. The separate financial increment is added below. There is no additional debt-running or timeout-shortfall penalty; actual interest and fees reduce net worth. This replaces an earlier calibration reward whose larger terminal failure penalty made delayed starvation preferable to an early crash.

An explicit bounded secondary financial objective adds U(W_next)-U(W_current), where U(W)=0.1 clip(W/2000,-1,1). Borrowing and repayment are wealth-neutral apart from charges. Its entire discounted contribution is bounded in absolute value by 0.2, so it cannot create an unbounded debt penalty or an unlimited wage-farming objective. It is linear in wealth within +/- $2000 and saturates outside; saturation is recorded and limits risk interpretations. This financial term changes the objective and is explicitly ablated in T09. It is distinct from policy-invariant potential shaping.

Dense feedback has no casino-directed target or visit bonus and is potential-based: F=gamma Phi(next)-Phi(current). The bounded potential combines net-worth progress, food, fuel, fatigue and distance to useful services. It is zero at all true terminals. Its discounted telescoping sum is -Phi(start) for a terminal episode, so cycling through sleep, borrowing and repayment cannot create extra terminal return through shaping alone. Loans are not income, debt payments are not a second income event, and there is no repeated visit bonus. The verifier checks the shaping identity and cash/debt conservation. Administrative censoring of continuing evaluation is not an environment terminal and is reported separately.

## Reflection and verification

The central difficulty was separating economic failure from navigation and accounting errors. Development checks found unsafe stops near traffic, late food planning, excessive bank trips, and a batch-axis error in the loan-count log; each was corrected before freezing the main study. All final evaluation actions and endpoint accounts are independently replayed by [audit_results.py](../src/audit_results.py); this complements intended-rule checks rather than proving optimality. Correct circle geometry is checked independently of a controller's apparent success. Development calibrations are retained under `runs/` and do not count as final policy comparisons. [verify.py](../src/verify.py) covers driving/coasting, pushing, starvation, absorbing terminals, individual loans, repayments, ownership blocking, checkpoint charges, meal timing, scenario splits, route balance and shaping. [analytic_checks.py](../src/analytic_checks.py) computes game expectations separately from learned outcomes.

## Results, video and reproduction

{status} The assignment does not require an optimal agent. Final outcomes are reported in [results.md](results.md), including unsuccessful policies. The numeric log is the source of truth: training and evaluation are headless; the renderer reads saved trajectories afterward. Commentary consists of deterministic English event receipts generated from those records, not the agent's internal reasoning. The replay shows individual microloans and total liabilities.

Use the [README](../README.md) setup instructions, run `python src/verify.py`, then `python src/evaluate.py --reference reserve --scenario-start 1 --episodes 10 --output runs/reference`. Render with `python src/render_replay.py runs/reference --scenario-id 1 --output runs/reference.mp4`. Learned-policy evaluation and replay commands are documented there too. [Course repository](https://github.com/InnoRL/IAEML-assignment).
"""
    a2 = f"""# Assignment 2: Learning and Evaluation

Iuliia Kurnaeva  
{now}

## Experimental setup

{status}

The agent selects economic service intentions and bounded continuous driving styles; a shared route controller executes real acceleration and steering. Its goal is to survive, clear both kinds of debt, and complete ownership. An episode terminates on ownership, collision, starvation or a finite deadline. Continuing-task evaluations report remaining active episodes as censored at 30 days. The [environment report](assignment1.md) defines mechanics and rewards. There are no economic demonstrations in the principal experiments. Preserved historical weights remain in the baseline archive and are not silently reused with incompatible observations.

The design separates algorithm, architecture, training horizon, scenario and checkpoint. Three main agents—Hybrid DSAC–MLP, hybrid PPO–MLP and hybrid PPO–GRU—are trained independently on five-day, ten-day and continuing tasks with five initializations each (45 runs). Eleven ten-day extensions add five runs each (55 runs): no curriculum, fixed market, decaying RND, final-payment-only contract, unavailable casino, recurrent DSAC, deterministic meal means, wider matched-mean meal portions, removal of bounded financial-progress reward, and removal of the derived navigation cue, and fixed discounting during the curriculum. All individual IDs and completion states are in [registry.csv](../experiments/registry.csv); all comparison definitions are in [EXPERIMENTS.md](../EXPERIMENTS.md). X01–X03 add 25 registered training arms for hidden casino regimes, reduced public quotes and an accelerated five-day payment calendar.

{COMMON}

## RL models and updates

The hybrid actor represents 26 service intentions and a two-dimensional tanh-squashed Gaussian driving style. Continuous log probability contributes for destination-bearing intentions; Wait, Hit and Stand have no continuous branch. This is a true hybrid probability distribution, rather than thresholding a third continuous control. Ineligible operations are masked before normalization. Actor inputs include the same public observation and eligibility mask for all principal algorithms. A derived direction/distance cue selects a nearby service from public resource and contract conditions, with no casino target or action override. This feature-engineering aid is disclosed and removed in T10; the original per-service directions remain available.

Hybrid DSAC uses two 128-unit hidden layers, twin critics with 24 return quantiles, a target critic and experience replay. Its actor maximizes the critic's expected soft return. The quantile critic models the return distribution, and the principal actor optimizes its mean. Discrete actions are summed exactly while the continuous branch uses a reparameterized sample. Twin distributions are selected by their mean rather than taking incompatible quantile-wise minima. Quantile Huber regression trains the critics. Separate continuous and discrete entropy temperatures start at 0.0006 and 0.0012 and are learned at 3e-4; critic Adam rate is 2.5e-4 and actor Adam rate is 5e-5, target update rate 0.008, gradient norm limit 5. Temperature log values are clipped to [-12,-2]. There are 32 minibatch updates of 256 examples per collection of eight decisions over 64 environments. Each intention has at most 32 active physical ticks; collection has variable physical duration. Replay capacity is 65536. Critic output initialization scale is 0.02. Both PPO and DSAC smoothly bound the Gaussian mean as 3*tanh(raw_mean/3); Gaussian noise retains full support over the squashed continuous action interval. Sources: [distributional SAC](https://arxiv.org/abs/2004.14547), [Hybrid SAC poster](https://staticctf.ubisoft.com/J3yJr34U2pZ2Ieem48Dwy9uqj5PNUQTn/6DcUzo72uIDbw5Piijo2AQ/2bcd7b560e172b5411cb1b86cf0dfa5b/AAAI_2020_FINAL.pdf), [quantile regression RL](https://arxiv.org/abs/1710.10044). 

PPO uses a 128 × 128 feature encoder with SiLU after both trunk layers, Gaussian/categorical actor heads and a scalar value head. The recurrent variant inserts a 128-unit GRU. The joint hybrid action likelihood enters a clipped policy objective with ratio range [0.8,1.2]. GAE uses physical gamma=0.999 raised to each intention duration and decision-level lambda=0.95; value-loss weight is 0.5, entropy coefficients are 0.003 for driving style and 0.01 for the discrete branch. The continuous entropy uses a reparameterized sample from the current tanh-Gaussian and is weighted by the summed probability of style-bearing intentions; categorical entropy is exact. Treating a fixed old action's negative log likelihood as differentiable entropy would give a zero expected score gradient and is not used. Adam rate is 2.5e-4 and gradient norm is clipped to 0.5. Each eight-decision rollout over 64 environments undergoes three epochs in environment minibatches of eight, preserving within-sequence temporal order. Hidden state resets at episode boundaries. [PPO paper](https://arxiv.org/abs/1707.06347).

The DSAC memory extension uses a learned 128-unit GRU over the most recent 32 raw observations, with raw observations stored once in replay and same-environment sequences gathered at update time, zeroed at reset boundaries and a target encoder. Padding is skipped. This finite-history representation differs from PPO's persistent recurrent state; conclusions must name that difference. An RND extension uses a fixed random target network and a trained predictor over lidar features. Its gamma-scaled intrinsic bonus has a fixed-task discounted bound of 0.05 and decays to zero by 75% of training; final selection and reporting always use the external task objective.

Each registered run receives 4194304 active physical environment transitions. Collection overshoot is at most 16383 transitions and is recorded. The first 20% starts at home with $200 principal, slower resource consumption and fixed market, with physical discounts 0.995/0.997/0.998/0.999 across four equal substages (T11 uses the final discount throughout). The next 30% uses $500 debt and mixed service/home starts, with home probability 25%, 50% and 75%. Food starts have low satiety, fuel starts have little fuel and casino starts are at opening; no action is forced. The last 50% uses the full $1000 task and normal home start. Validation always uses final conditions; replay clears at curriculum transitions. Equal physical experience is the budget, while compute and optimizer counts differ.

The shared driver follows the selected service destination under actual bicycle dynamics, time, costs and traffic. It does not choose an economic operation or provide an economic demonstration. An intention lasts at most 32 physical ticks, executes at most once and ends on service completion or game feedback. Wait lasts at most eight ticks and can end earlier on feedback. The return sums discounted physical rewards; continuation discount is physical_gamma**actual_duration. Padding consumes no simulated time or resources. The observation contains 153 public physical/economic features and 26 previous-intention indicators. Finite-history DSAC retains 32 intentions of variable physical duration; PPO retains persistent GRU state. Low-level raw-driving pilots remain engineering history, rather than the primary comparison.

Development controller checks covered 900 directed service-pair routes: 779 arrivals, 23 collisions and 98 unfinished under the selected forecast. Ninety unfinished routes were departures from the terminal garage. These diagnostics disclose residual navigation limitations and are not full-game success estimates. The longer safety forecast remains an explicit frozen intervention. [Route benchmark](../experiments/navigation_h24_hold4.json).

## Computational optimization and preservation

JAX compiles a 64-environment × 64-step headless collector and batched learner updates. Static route fields are computed once, not on every tick. Vectorization covers environments and candidate actions, and `lax.scan` handles time. Compilation caches reuse matching shapes/configurations across initializations and validation checkpoints; actor weights remain dynamic inputs, avoiding repeated compilation of a constant policy. The local verified GPU is an RTX 3080 Ti Laptop GPU with 16 GiB. Runtime versions and device information are frozen alongside sources. Performance numbers are read from actual run logs, not estimated from the GPU name.

Compressed training trajectories retain requested/executed actions, resources, debts, rewards, events, entropies and loan receipts. Validation and test trajectories additionally retain exact public observation vectors for policy probes. Full observation vectors are omitted from training logs to reduce storage, while replay and resumable states retain what learning needs. Every final evaluation is replayed from its reset with the exact saved policy. Action choices, controls, numeric state fields and endpoints must match, and only audited evaluations enter this report. Independent rule tests cover the simulator, and exact replay checks the evaluation records. Full resumable checkpoints store actor, critic, target networks, optimizers, temperatures, replay, environment states, hidden state, random keys and counters. Source hashes must match to resume. Policy-only NPZ snapshots avoid pickle for inference. Save points include initial, curriculum boundaries, periodically selected/last and final weights. The selected model follows the declared validation ordering: mean discounted external return, ownership, survival, remaining debt and completion time. Final test cases never select weights. Calibration uses initializations 101/102, execution checks use 9091/9092, and the main initialization bank 201-205 is kept separate. Early driving substages use discounts 0.97, 0.985, 0.995 and 0.997, with 0.999 thereafter and throughout evaluation; T11 holds the discount at 0.999. This temporal curriculum amplifies short-task progress while the agent learns physical skills. DSAC bootstrapping and environment potential shaping always use the same active discount.

## Results and reflection

{plot1}

{plot2}

Short runs test execution and checkpoint persistence. An initial full-budget engineering pilot failed to learn: all deterministic validation runs ended in starvation or collision and no work was completed. A sampled-policy diagnostic also failed. Calibration reduced DSAC temperatures, increased bounded navigation shaping before an explicit bounded secondary financial-progress objective was added, removed casino destinations from shaping, and aligned checkpoint selection with external return. A subsequent physics audit removed instantaneous stopping on Wait, and the accessible-start curriculum was made genuinely reachable by including near-ownership states. A subsequent logging audit corrected the batched dealer up-card axis; this affected receipts, not the environment's card state or observations. A PPO audit corrected the continuous entropy gradient estimator and supplied the second trunk activation. Short tutorial routes remain in the expanding curriculum. A subsequent reward audit found that discounting a large terminal failure penalty favored delayed starvation over early crashes. The current failure-neutral base return removes that timing preference algebraically. Entropy recovery, bounded Gaussian means, smaller critic initialization and broader safe tutorial approaches were calibrated together. These changes were evaluated together. A failure to learn may reflect limited experience, long navigation/service chains or exploration rather than infeasibility; development reference strategies and analytical accounting help separate these possibilities. Casino avoidance can be a valid consequence of negative expected return. Casino participation near a hard deadline can improve a threshold-success probability while decreasing expected money. 

The complete observed results and missing comparisons are in [results.md](results.md). Unsupported conclusions remain explicitly pending. Main scientific comparisons need all five independent initializations; an early single successful run is not sufficient. [Research report](research.md) discusses uncertainty and practical scope.

## Checkpoints, video and local reproduction

Run `python src/run_suite.py --phase all --full-frozen-suite` after setup to train the registered suite and evaluate saved selections. Use `--run-id` to isolate one registered model. `python src/build_reports.py` rebuilds these reports from artifacts. Each learned checkpoint is stored under `artifacts/training/<run_id>/selected.npz`. Evaluation reads it using `python src/evaluate.py --checkpoint <path> --scenario-start 2000001 --episodes 256 --output runs/reproduction`. Render a recorded scenario with `python src/render_replay.py runs/reproduction --scenario-id 2000001 --output runs/reproduction.mp4`. Videos reproduce recorded data and never run during optimization. The [README](../README.md) gives exact setup and resume examples.


"""
    research = f"""# Financial Deadlines, Survival Costs and Voluntary Risk in a Driving Task

Iuliia Kurnaeva  
{now}

## Abstract and current evidence

We implement a JAX driving task coupling navigation, work, sleep, food, fuel and two classes of debt. The driver has $100 of initial liquidity and must repay a $1000 car obligation while maintaining survival resources. A registered study compares hybrid distributional SAC and PPO, feedforward and recurrent representations, and five-day, ten-day and continuing tasks. Further comparisons separate training interventions from frozen-policy stress tests. Negative-expectation games offer faster, uncertain cash flows. The objective is held constant rather than directly rewarding gambling.

{status} 

## Motivation and research questions

Expected money, probability of repaying a deadline-bound obligation, and speed of repayment are different criteria. An action with negative financial expectation may still increase the chance of crossing a payment threshold before time runs out. Resource costs complicate the decision: keeping an emergency cash reserve can delay debt repayment but prevent a high-interest liquidity spiral. Cheap food saves travel time yet is expensive per unit of satiety. These tradeoffs motivate measurable questions: which algorithms learn ownership, how horizon alters risk, whether memory helps under predictive uncertainty, and which failures arise from economic planning versus vehicle control.

The project is a synthetic stress-test environment. Possible practical uses include evaluating planning methods under liquidity constraints, testing robustness to operating-cost shocks, studying rare-failure reporting and comparing exploration in mixed action spaces. It is not calibrated to household finances, taxi businesses, real casinos or human dependence. Its results cannot directly recommend lending or gambling behavior.

## Task construction and development record

The starting design attached economic choices to a required vehicular environment. Earlier iterations mixed emotional variables, imposed gambling impulses and changing reward definitions, making behavioral interpretations difficult. The current study removes those mechanisms, retains one external objective, separates personal debt from car debt and assigns every experimental condition an ID. Historical artifacts are preserved with a checksum manifest and are not pooled with current results.

Development uses only scenario IDs 1–10. Accounting tests exposed the need to distinguish loan proceeds from income and interest/fees from repaid principal. Service timing must consume hunger before a meal can rescue the driver. Navigation checks enforce the sum of two unit-circle radii and distinguish safe parking from merely entering an interaction radius. Feasibility pilots compare nearby low-restoration meals against distant larger portions and batch bank payments to reduce travel. The salary was calibrated from a provisional $145 to $175 before the frozen study. Earlier calibration outcomes under different geometry or salaries are engineering history, not interchangeable estimates of the final environment.

## Formal objective and analytical results

Let cash be C, outstanding car debt D_c, personal debt D_p and reserved stake E. Financial net worth is W=C+E-D_c-D_p. Loan disbursement adds equal cash and principal. Repayment subtracts equal cash and liability; neither creates wealth. Interest and late fees lower W. A successfully delivered shift raises W, while purchases and casino losses lower it. This accounting prevents credit from becoming a reward loophole.

The external reward includes a bounded secondary wealth-utility increment as well as terminal ownership/survival and time costs. Its clipping can affect behavior outside the linear wealth range; T09 measures its contribution. The external reward and potential are defined in [Assignment 1](assignment1.md). The discounted running cost is bounded by 0.1. The base return of every unsuccessful finite trajectory is -0.1, independent of duration, removing a pure failure-delay incentive. Potential shaping telescopes to -Phi(start) at true terminal states. These are mathematical statements about the specified reward. In contrast, the held-out empirical tail is an estimate with uncertainty.

Slots return pM-1=-0.290909 dollars per dollar wagered. An exact dynamic program enumerating replacement-card ranks and optimal hit/stand decisions yields blackjack EV={bj:.8f} per dollar under the implemented rules. Even the favorable latent slot regime has negative expectation (-0.054545 per dollar). The games have negative expected income. A wager can still raise the probability of reaching a fixed cash threshold in some states.

Ratios of expected food price to raw portion are 100, 100 and about 53.85 dollars per satiety unit for diner, cafe and restaurant. These are planning benchmarks, not a proof that the restaurant is optimal. Travel, clipping, remaining lifetime, service time, fuel, missed shifts and cash on hand can reverse the preference. Higher-variance portions have the same nominal means but a different starvation tail. Comparisons against an expected-value benchmark must condition on the actual task objective and opportunity costs.

## Experimental design and reproducibility

{COMMON}

The complete comparison matrix is [EXPERIMENTS.md](../EXPERIMENTS.md). The [policy catalog and crossed uncertainty analysis](study_analysis.md) and [calibration catalog](calibration_catalog.json) keep main evidence separate from engineering history. M01/M02/H01 comprise 45 main runs. T01–T11 add 55 training runs, changing one declared factor against the ten-day DSAC reference. H02 and F01–F12 reuse frozen weights to change horizon, control strength/noise, driving guard, initial fuel/satiety/cash, wages, fuel price, initial personal debt, food variance and service identity. No driving-controller intervention is silently described as the original learned policy. X01–X03 have implementation flags for latent casino regimes, reduced public quotes and an accelerated calendar, and 25 registered matched training arms.

P01 measures the first eligible voluntary wagering decision after a loss, reports its denominator and censored follow-ups, and probes public-event sensitivity at matched cash/time/resources. P02 compares recurrent action distributions with preserved versus reset memory on recorded observations. Synthetic matched probes are labeled as such. Differences do not reveal internal verbal thoughts or establish psychological motives. Near-zero gambling participation leaves loss-chasing undefined rather than evidence of zero dependence.

Each artifact retains source/config hashes, scenario IDs, checkpoint hash and requested/executed decisions. Scenario RNG streams are separated from policy sampling; daily prices use absolute day, portions use venue and purchase ordinal, games use game stream and resolved-round ordinal, and traffic uses its own initialization stream. Different policies need not consume the same outcome sequence at the same time; the matching convention is explicit. Sampling the trained stochastic policy is primary, using a separate policy RNG keyed by scenario and absolute tick. Argmax/mean extraction is a separately labeled diagnostic. Training transitions and evaluation episodes are different experimental units.

## Observed results

{plot1}

{table([r for r in results if r['case'] == 'standard'])}

The machine-readable source of every row is `artifacts/evaluation/<run_id>/<case>/evaluation.json`, with endpoint arrays, uncertainty intervals and complete compressed traces. [results.md](results.md) lists all completed cases and completion status. Until the suite is complete, partial algorithm rankings and differences between horizons are provisional. Selection is validation-only; evaluating many frozen stress cases does not authorize retuning on them and reusing the same final bank as untouched evidence.

## Interpretation, limitations and contribution

The reusable contribution currently established is the auditable environment, hybrid-learning implementation, separated accounting, registered condition matrix and preserved trajectories. A quantitative algorithmic or behavioral contribution requires completed replicated comparisons against relevant existing work. General human findings such as greater risk under pressure cannot be claimed as new discoveries from this simulation alone. A conference-level claim would additionally need broader baselines, learning-budget sensitivity, stronger external validity and a literature-positioned novelty argument.

Five training repetitions provide a limited estimate of optimization variation, and 256 scenarios give imprecise extreme tails. Bootstrap intervals can degenerate when every outcome is identical. The continuing evaluation right-censors episodes that are still active at the time limit. Mean return and CVaR summarize different parts of the return distribution.

## Reproduce and extend

The [README](../README.md) specifies setup, exact run IDs, source freeze, checkpoints, replay and report generation. Execute the full registered suite with `python src/run_suite.py --phase all --full-frozen-suite`. Preserve `experiments/freeze.json`, registry, dependency record, complete checkpoints and trajectory chunks together. Increasing scenario count requires validating disjoint banks. Any post-freeze environment/reward change requires a new protocol and separated results, rather than overwriting an experiment.

## References

- [InnoRL IAEML course starter](https://github.com/InnoRL/IAEML-assignment).
- [Delalleau et al., Hybrid SAC poster: Discrete and Continuous Action Representation for Practical RL in Video Games](https://staticctf.ubisoft.com/J3yJr34U2pZ2Ieem48Dwy9uqj5PNUQTn/6DcUzo72uIDbw5Piijo2AQ/2bcd7b560e172b5411cb1b86cf0dfa5b/AAAI_2020_FINAL.pdf).
- [Ma et al., DSAC: Distributional Soft Actor-Critic for Risk-Sensitive Reinforcement Learning](https://arxiv.org/abs/2004.14547).
- [Burda et al., Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894).
- [Ng, Harada and Russell, Policy Invariance under Reward Transformations](https://aima.cs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
- [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- [Dabney et al., Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044).
"""
    for name, content in (("assignment1.md", a1), ("assignment2.md", a2), ("research.md", research)):
        (target/f"generated_legacy_{name}").write_text(content, encoding="utf-8")
    progress = f"# Recorded experiment results\n\n{status}\n\n"
    progress += "| Registered run | Status | Completed training |\n|---|---|---|\n"
    progress += chr(10).join(f"| {r['run_id']} | {r['status']} | {str((ROOT/'artifacts/training'/r['run_id']/'complete.json').exists())} |" for r in rows)
    progress += "\n\n## Completed frozen evaluations\n\n"+table(results)+"\n"
    (target/"generated_legacy_results.md").write_text(progress, encoding="utf-8")
    print(status, flush=True)


if __name__ == "__main__":
    main()
