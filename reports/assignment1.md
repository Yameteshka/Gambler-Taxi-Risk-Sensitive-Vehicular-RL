# Assignment 1: Environment

Iuliia Kurnaeva  
23 November 2026

## Design motivation

The project began with a question about risk under debt: when does a driver give up relatively predictable earnings for a chance at an immediate payout? A taxi made this a spatial and temporal problem. A trip to the casino took time, and a late session could leave the driver fatigued or cost the next shift. Repeated wagers could therefore affect the campaign long after the money from a particular round had been settled. [Prototype design](../baseline/reports/assignment1.md) and [early experiments](../baseline/reports/debt_research.md)

I introduced mood, tilt, anxiety and habit to test particular forms of carryover. Mood responded to wins, losses and other events, and low mood reduced traction. Tilt rose after gambling outcomes and increased the available steering angle. Anxiety persisted with unpaid debt, rose when borrowing and slowed the steering response. Habit accumulated across rounds and decayed faster during sleep. It could also contribute to an optional impulse rule that turned a departure into a wager. These were modeling assumptions about persistence and feedback. I logged requested and executed actions separately and disabled the override in the voluntary-wager analysis, since an executed wager alone could not show that the policy chose to continue. [Prototype implementation](../baseline/src/env/gamblers_taxi.py) and [voluntary wagering analysis](../baseline/reports/debt_research.md#53-voluntary-continuation-and-loss-specific-response)

## From a gambling prototype to resource planning

A large payout could dominate a short campaign, leaving little opportunity to study how the driver balanced work, rest and borrowing. I extended the horizon, restricted betting hours and added a cooldown shared by both games. For the resource-management study I removed the psychological state and the impulse override. This reduced the number of hand-specified effects connecting gambling outcomes to later behavior. I kept fatigue because it creates a direct trade-off between the time spent sleeping and the ability to drive. I replaced hiring and dismissal with daily shift windows, so a missed shift still cost income without adding a separate job-search task. [Experiment history](study/experiments.md) and [design problems and fixes](study/problems_and_fixes.md#environment-and-reward-design)

The redesigned driver starts with $100 in spending money. Food and fuel compete with debt repayment for that balance. A personal loan provides liquidity and creates a separate interest-bearing liability. The garage requires the car balance and all personal loans to be cleared. The three food venues have different prices, portions and service times, which are visible to the policy. Empty fuel limits the vehicle to pushing speed and increases resource consumption. Zero satiety terminates the episode. Together these rules make the order of ordinary decisions consequential, even in a run with no gambling. [Environment rules](study/environment.md), [public food menu](public_menu.md) and [environment diagram](../media/figures/environment_model.svg)

## Vehicle model and execution

The environment is a JAX subclass of `BaseEnv` in [resource_taxi.py](../src/env/resource_taxi.py). Its physical actions are continuous acceleration and steering. I chose a kinematic bicycle model to represent turning radius and braking distance with a small, fixed-size state update. Rolling resistance and speed-dependent drag slow the vehicle, while fatigue reduces braking and steering response. Lidar provides local obstacle information. The policy also sees relative service positions and obstacle-aware road distances. A swept collision test covers contact between ticks. The shared controller executes the destination selected by the policy, with parking, travel time and fuel use governed by the simulator. [Course starter](https://github.com/InnoRL/IAEML-assignment), [model and observations](study/environment.md) and [environment verification](study/data/diagnostics/environment_verification.json)

The initial service interface made some strategic choices difficult to execute. In 60 food-trip checks from home and work, no single 32-tick decision delivered a meal. Maintaining the destination across decisions delivered a meal in all 60. I extended the maximum duration of a service decision to 128 physical ticks and returned control on completion, refusal or termination. This gave a food decision enough time to take effect. Subsequent starvation cases required checking the action trace for skipped meals, late requests and depleted cash. [Food-trip tests](../experiments/food_execution_diagnosis.json) and [failure diagnosis](training_diagnosis.md)

## Reward design

The earlier combination of discounting and a large terminal penalty gave later failures a higher base return than comparable early failures. I changed the terminal penalty and per-tick cost so every finite unsuccessful trajectory has a discounted base return of −0.1. Ownership pays +1 plus a cash-reserve bonus of at most 0.05, and discounting favors earlier completion when the financial outcome is otherwise the same. A separate bounded net-worth term counts cash and liabilities together, so taking a loan adds equal amounts on both sides of the balance sheet before interest. I left potential shaping and targeted navigation cues disabled in the reported study. [Reward derivation](reward_audit.md), [reward and accounting rules](study/environment.md) and [final outcome checks](study/final_dsac_gru.md#reward-hacking)

## Demonstration and reproduction

The [environment demonstration](../media/video/environment-demonstration.mp4) records a reference controller completing the $1,000 campaign. The corresponding [event log](../media/video/environment-demonstration.events.json) records service use and repayment. With the dependencies installed, the recorded trajectory can be rendered from the repository root.

```bash
python src/render_replay.py reports/study/data/replays/environment --scenario-id 2 --output runs/review-environment.mp4 --stride 4
```

[Local setup and reproduction](study/reproduce.md), [Assignment 2](assignment2.md) and [study index](study/README.md)
