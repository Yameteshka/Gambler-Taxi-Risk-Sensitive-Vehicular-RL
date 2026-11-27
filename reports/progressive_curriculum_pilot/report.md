# Gradual DSAC pilot: first-lesson stop

## Results after 614,296 physical steps

The last actor-only lesson check (16 validation scenarios, $50 principal, balanced service contexts, reduced hunger, no added exploration) achieved 14 ownership episodes, one collision and one starvation. It completed no paid work, meals, sleep, fuel purchases or casino games in that check. Some lesson contexts start with the car debt already settled. 

The last full-task check (64 validation scenarios, standard home start, $500 principal, dynamic market) at step 549,024 had zero ownership, 39 starvation deaths and 25 collisions. The three recorded full-task validations all had zero ownership. The latest full-task actor did complete 38 paid shifts across 64 episodes, but no meals, sleep, refueling or casino games. 

Exploratory training exposure included 144 paid shifts, 563 meals, 218 sleeps, 388 refills, 237 slot games, 206 blackjack hands, 765 loans, 812 car-payment events and 364 microloan-payment events. These events occurred during training with an 80% exploration mixture.

## Interpretation and limitation

Training stopped before the stages that increased debt or annealed exploration.

The initial task and its advancement criterion are poorly aligned. Starting with $100 cash and only $50 car debt allows quick ownership without earning or sustaining a long life cycle. The guard then demands discretionary work, meals, sleep and fuel purchases that need not help solve that easy task. The observed policy improves easy ownership while failing those criteria. 

A meaningful follow-up must make the skills useful in controlled training contexts and test them where needed, without adding casino bonuses, economic action labels or reporting random exploration as mastery. Earning-required tasks must be checked before a full four-model launch. This observation does not by itself establish that curriculum design explains every DSAC failure; critic, entropy and variable-duration-action diagnostics remain relevant.

## Evidence

- [Machine-readable gate, validation and exposure history](summary.json).

No final test bank was consumed by this pilot. These are development results from one initialization.
