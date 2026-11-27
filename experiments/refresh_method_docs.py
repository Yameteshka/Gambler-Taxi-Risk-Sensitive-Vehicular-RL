"""Align current method documentation; archived pilot sources remain untouched."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
method = """## Shared driving and learned economic decisions

The primary policy selects 26 explicit intentions: wait, work, sleep, three food venues, two refueling quantities, three loan amounts, six repayment amounts, three slot stakes, three blackjack stakes, hit, stand and ownership. A common route controller converts the selected destination and two Gaussian style parameters into real acceleration and steering. Speed factors range from 0.8 to 1.0 and steering factors from 0.9 to 1.1. Physics, fuel, hunger, fatigue, traffic, collisions and time remain active. The controller does not select a financial strategy, a restaurant, a casino or a loan. No economic demonstrations are used.

An intention runs for at most 32 physical ticks and can execute its requested operation once. Service completion or game feedback returns control immediately. Wait lasts at most eight ticks and ends earlier when an ongoing service finishes. Blackjack hit/stand decisions return control after one card action. Padded storage slots freeze the simulation and do not count as experience. Each decision reward sums its physically discounted rewards; bootstrapping uses gamma raised to its actual duration. Physical gamma is 0.999 in the final task. The policy sees 153 public environment features plus the previous-intention one-hot vector, for 179 features. A persistent PPO GRU updates per decision; finite-history DSAC retains 32 decisions, spanning at most 1024 physical ticks.

Every run receives 4,194,304 active physical transitions, with final overshoot below 16,384 because collection uses 64 environments and eight decisions. DSAC makes 32 updates of 256 examples per collection; PPO makes three epochs in environment minibatches of eight. The initial 20% of the economic curriculum starts at home with $200 principal, slower resource consumption and fixed prices. Its four equal substages use physical discounts 0.995/0.997/0.998/0.999, except T11. The next 30% uses $500 debt and mixed home/service starts. The final 50% uses the complete $1000 task at home. Validation always uses the complete task. Raw-steering driving-school pilots are preserved as calibration history and are not the primary method.

The controller's development benchmark covers 900 directed service-pair routes on scenarios 1–10. The selected 24-tick safety forecast with a four-tick held command achieved 779 arrivals, 23 collisions and 98 unfinished routes. Ninety unfinished routes were departures from the garage, where successful campaigns already terminate. This is a route diagnostic, not an ownership result or a claim of collision-free driving. The longer-forecast frozen intervention is reported separately.

The 125 registered rows remain the experiment matrix. Main initializations are 201–205, calibration uses 101/102, and execution checks use 9091/9092. Development scenarios 1–10, training 11–10010, validation 1000001–1000064 and final scenarios 2000001–2000256 are disjoint. Test-bank evaluation begins after the final source/configuration freeze. All training and evaluation are headless; videos are reconstructed afterward from saved physical trajectories and decisions.
"""

for filename in ('README.md','EXPERIMENTS.md','PLAN.md'):
    path=ROOT/filename
    text=path.read_text(encoding='utf-8')
    # Old revision appendices mix the superseded raw-steering curriculum with
    # the restored primary method. Full historical copies remain in pilot sources.
    markers=[text.find('\n## '+s) for s in ('Transfer and budget calibration','Final initialization separation','Reward and execution audit','Policy-gradient and trajectory checks')]
    markers=[x for x in markers if x>=0]
    if markers:text=text[:min(markers)].rstrip()+'\n'
    text=text.replace('Original continuous dynamics, no assistance','Shared route controller; real continuous dynamics')
    text=text.replace('days 2, 3, and 5','days 2, 4, and 5')
    text=text.replace('python src/train.py','python src/strategy_training.py')
    if filename=='README.md':
        text=text.replace('python src/analytic_checks.py','python src/analytic_checks.py\npython src/verify_strategy.py')
    text+='\n'+method
    path.write_text(text,encoding='utf-8')

path=ROOT/'src/build_reports.py'
text=path.read_text(encoding='utf-8')
text=text.replace('The agent controls both physical driving and service decisions.',
    'The agent selects economic service intentions and bounded continuous driving styles; a shared route controller executes real acceleration and steering.')
text=text.replace('The hybrid actor represents a categorical choice of 25 operations and a two-dimensional tanh-squashed Gaussian control. Continuous log probability contributes only when the selected operation is Drive.',
    'The hybrid actor represents 26 service intentions and a two-dimensional tanh-squashed Gaussian driving style. Continuous log probability contributes for destination-bearing intentions; Wait, Hit and Stand have no continuous branch.')
text=text.replace('There are 25 categorical choices.',
    'There are 25 physical categorical choices. The learned service interface maps 26 intentions onto these physical operations and a shared route controller, with 179 public input features.')
text=text.replace('There are 32 minibatch updates of 256 examples per 4096-transition collection chunk.',
    'There are 32 minibatch updates of 256 examples per collection of eight decisions over 64 environments. Each intention has at most 32 active physical ticks; collection has variable physical duration.')
text=text.replace('GAE uses gamma=0.999 and lambda=0.95;',
    'GAE uses physical gamma=0.999 raised to each intention duration and decision-level lambda=0.95;')
text=text.replace('entropy coefficients are 0.003 for driving','entropy coefficients are 0.003 for driving style')
text=text.replace('weighted by its current Drive probability','weighted by the summed probability of style-bearing intentions')
text=text.replace('Each 64-step rollout over 64 environments','Each eight-decision rollout over 64 environments')
start=text.index('Each registered run receives 4194304 physical environment transitions.')
end=text.index('\n\n## Computational optimization',start)
replacement="""Each registered run receives 4194304 active physical environment transitions. Collection overshoot is at most 16383 transitions and is recorded. The first 20% starts at home with $200 principal, slower resource consumption and fixed market, with physical discounts 0.995/0.997/0.998/0.999 across four equal substages (T11 uses the final discount throughout). The next 30% uses $500 debt and mixed service/home starts, with home probability 25%, 50% and 75%. Food starts have low satiety, fuel starts have little fuel and casino starts are at opening; no action is forced. The last 50% uses the full $1000 task and normal home start. Validation always uses final conditions; replay clears at curriculum transitions. Equal physical experience is the budget, while compute and optimizer counts differ.

The shared driver follows the selected service destination under actual bicycle dynamics, time, costs and traffic. It does not choose an economic operation or provide an economic demonstration. An intention lasts at most 32 physical ticks, executes at most once and ends on service completion or game feedback. Wait lasts at most eight ticks and can end earlier on feedback. The return sums discounted physical rewards; continuation discount is physical_gamma**actual_duration. Padding consumes no simulated time or resources. The observation contains 153 public physical/economic features and 26 previous-intention indicators. Finite-history DSAC retains 32 intentions of variable physical duration; PPO retains persistent GRU state. Low-level raw-driving pilots remain engineering history, rather than the primary comparison.

Development controller checks covered 900 directed service-pair routes: 779 arrivals, 23 collisions and 98 unfinished under the selected forecast. Ninety unfinished routes were departures from the terminal garage. These diagnostics disclose residual navigation limitations and are not full-game success estimates. The longer safety forecast remains an explicit frozen intervention. [Route benchmark](../experiments/navigation_h24_hold4.json)."""
text=text[:start]+replacement+text[end:]
path.write_text(text,encoding='utf-8')
print('Current method documentation aligned; numerical outcomes remain evidence-driven.')
