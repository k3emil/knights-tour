# Results

Machine: MacBook Pro, Intel i7-9750H (6 physical / 12 logical cores), Python 3.12.

## Step 2 — random-legal-move baseline (6x6, all 36 starts)

Command: `python baseline_random.py --episodes 10000 --board-size 6`

| metric | value |
|---|---|
| episodes | 10,000 |
| full tours | 0 (0.000%) |
| squares covered | mean 20.10, max 35, min 4 |
| elapsed | 27.0 s |

Reading: picking uniformly among legal knight moves never completes a 6x6 tour in 10,000
tries, and gets about 20 of 36 squares on average. This is the bar the trained policy has
to beat.

It also confirms two design decisions. The action mask works (no illegal moves were ever
attempted, and episodes end when the knight is trapped). And a completion-only reward
would give almost no learning signal, which is why the environment pays +1 per newly
visited square.

## Step 4 — training (6x6, MaskablePPO + MlpPolicy)

Command: `python train.py --steps 3000000 --board-size 6 --n-envs 6 --model-out model_ent03.zip`
Log: `train_ent03.log`

| setting | value |
|---|---|
| algorithm | MaskablePPO (sb3-contrib 2.3.2) |
| policy | MlpPolicy, `net_arch=[128, 128]` |
| observation | 3 planes of 6x6, flattened to 108 values |
| reward | +1 per new square, +10 on completing 36 |
| entropy bonus | `ent_coef=0.03` |
| parallel envs | 6 (SubprocVecEnv) |
| timesteps | 3,000,000 |

| result | value |
|---|---|
| wall clock | 19 min 09 s |
| throughput | 2,610 steps/s |
| best rolling success rate | 1.000 |
| final `entropy_loss` | -0.027 |
| saved model | `model_ent03.zip`, 786 KB |

Learning curve, from the log: mean coverage passed 31 of 36 within the first 150k steps
(baseline 20.10), reached ~36 with a 98% rolling success rate by about 1.15M steps, and
held a 100% rolling success rate thereafter. Training had effectively converged by
~1.5M steps, so the second half of the run added little.

## Step 5 — evaluation from every start square

Command: `python evaluate.py --model model_ent03.zip --board-size 6`
Log: `eval_ent03_deterministic.log`

Single shot per start, deterministic (argmax), no retries.

| metric | value |
|---|---|
| starts solved | **36 / 36 (100%)** |
| squares covered | mean 36.00, min 36, max 36 |
| elapsed, all 36 starts | 0.82 s |

All 36 tours were verified independently: every square numbered exactly once, and every
consecutive pair a legal knight move.

## Comparison (6x6)

| method | starts solved | notes |
|---|---|---|
| random legal moves | 0 / 36 | mean coverage 20.10 |
| RL, MaskablePPO + MLP | 36 / 36 | learned from reward only, no heuristic given |
| Warnsdorff's rule | 36 / 36 | hand-coded heuristic, tie-break = furthest from centre (the repo script's first-minimum tie-break gets 35 / 36) |

Speed comparison is deliberately out of scope for now.

## The finished policy is deterministic in practice

Verified across all 36 starts with 5 repeats each: exactly one distinct path per start.
That is expected for argmax with frozen weights.

More surprising, sampling barely differs. 20 sampled runs from start (1, 2) gave 1
distinct path and 20 / 20 full tours, because the policy is very confident — the top
action averages 0.998 probability and never drops below 0.949 across the 35 moves. So the
entropy bonus kept enough exploration *during training* to find good tours, while the
finished policy is effectively deterministic, behaving like a lookup table of 36 fixed
tours.

## It did not rediscover Warnsdorff's rule

Every start was played twice, once by the policy and once by Warnsdorff, and the paths
compared square by square against both tie-break variants:

| | vs repo tie-break | vs periphery tie-break |
|---|---|---|
| identical full path | 0 / 36 | 0 / 36 |
| same very first move | 23 / 36 | 22 / 36 |
| mean per-move agreement | 15.3% | 15.1% |
| median move where paths diverge | 3 | 4 |

The agent reaches the same outcome as the heuristic, 36 / 36, by its own route. The two
agree on the opening move about two-thirds of the time, then diverge within a few moves.
The shared instinct is only the broad one of not stranding squares; the specific "fewest
onward moves" rule is not what the network learned.

Caveat on reading this: identical paths were never likely, since a single start admits a
huge number of valid tours. The low per-move agreement is the meaningful figure, not the
zero exact matches.

## Hardest start on the board

Start (1, 2) is worth singling out, because it is the one the policy has least room for
error on. Corner (0, 0) has only two knight-neighbours, (1, 2) and (2, 1), so a tour
beginning on (1, 2) must take the corner either immediately as its second square or right
at the end as its last — anything else strands it. Both routes were verified to exist:

- taking (0, 0) as the 2nd square: tour found, ends at (2, 2)
- refusing (0, 0) as the 2nd square: tour found, ends at (0, 0)

There are 8 such corner-neighbour starts, two per corner: (1,2), (2,1), (2,4), (1,3),
(3,1), (4,2), (4,3), (3,4). The trained policy handles all of them.

## Deliberately not done

Rewarding low-degree squares. That is Warnsdorff's rule written into the reward function,
which would defeat the point of the experiment.
