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

Reading: picking uniformly among legal knight moves never completes a 6x6 tour in
10,000 tries, and gets about 20 of 36 squares on average. This is the bar the trained
policy has to beat.

It also confirms two design decisions. The action mask works (no illegal moves were
ever attempted, and episodes end when the knight is trapped). And a completion-only
reward would give almost no learning signal, which is why the environment pays +1 per
newly visited square.

## Step 4 — training (6x6, MaskablePPO + MlpPolicy)

Command: `python train.py --steps 3000000 --board-size 6 --n-envs 6`
Log: `train_mlp.log`

| setting | value |
|---|---|
| algorithm | MaskablePPO (sb3-contrib 2.3.2) |
| policy | MlpPolicy, `net_arch=[128, 128]` |
| observation | 3 planes of 6x6, flattened to 108 values |
| reward | +1 per new square, +10 on completing 36 |
| parallel envs | 6 (SubprocVecEnv) |
| timesteps | 3,000,000 |

| result | value |
|---|---|
| wall clock | 26 min 13 s |
| throughput | 1,907 steps/s |
| best rolling success rate | 1.000 |
| saved model | `model.zip`, 786 KB |

Learning curve, from the log: mean coverage was already 31.2 of 36 at 132k steps
(baseline 20.10), reached 35.9 by ~1.15M steps with a 98% rolling success rate, and
touched a 100% rolling success rate before the run ended.

## Step 5 — evaluation from every start square

Commands: `python evaluate.py --model model.zip --board-size 6`
and the same with `--attempts 10`. Logs: `eval_mlp_deterministic.log`,
`eval_mlp_sampled10.log`.

| metric | deterministic | 10 sampled attempts |
|---|---|---|
| starts solved | 35 / 36 (97.2%) | 35 / 36 (97.2%) |
| squares covered | mean 35.97, min 35, max 36 | mean 35.97, min 35, max 36 |
| unsolved starts | (1, 2) | (1, 2) |
| elapsed, all 36 starts | 1.38 s | 1.57 s |

The example tour printed by the evaluator was checked independently: all 36 squares are
numbered exactly once and every consecutive pair is a legal knight move.

Two observations:

- Retrying with sampled actions changed nothing. The policy's entropy collapsed during
  training (`entropy_loss` ended at -0.011), so sampling is effectively the same as
  argmax. Keeping some exploration at the end, by lowering `ent_coef` more slowly or
  stopping earlier, would make retries a genuine second chance.
- The single failure, start (1, 2), reaches 35 of 36 squares. A tour does exist from
  there (all 36 starts were verified solvable before training), so this is a policy gap,
  not an impossible board.

## Comparison so far (6x6)

| method | starts solved | notes |
|---|---|---|
| random legal moves | 0 / 36 | mean coverage 20.10 |
| RL, MaskablePPO + MLP | 35 / 36 | learned from reward only, no heuristic given |
| Warnsdorff's rule | 36 / 36 | hand-coded heuristic, tie-break = furthest from centre (the repo script's first-minimum tie-break gets 35 / 36) |

Speed comparison is deliberately out of scope for now.

## Step 5b — why start (1, 2) fails

Two questions were open after Step 5: does sampling help once entropy is lower, and is
the single failure a policy gap or something structural. Both were tested against the
saved checkpoints instead of retraining. Log: `eval_checkpoint_sweep.log`.

### Sampled retries do help — just not at the end of training

| checkpoint | deterministic | 20 sampled attempts |
|---|---|---|
| 600k | 35 / 36 | 35 / 36 |
| 1.0M | 34 / 36 | 35 / 36 |
| 1.2M | 33 / 36 | 35 / 36 |
| 1.6M | 35 / 36 | 35 / 36 |
| 2.0M | 35 / 36 | 35 / 36 |
| 2.5M | 35 / 36 | 35 / 36 |
| 3.0M (final) | 35 / 36 | 35 / 36 |

At 1.0M and 1.2M steps, retries recover starts that argmax misses (34 -> 35 and
33 -> 35). That confirms the entropy diagnosis: while the policy is still stochastic,
sampling is a genuine second chance. By 1.6M the entropy has collapsed and sampling
becomes indistinguishable from argmax, so retries stop adding anything.

Note the non-monotonic dip at 1.0M-1.2M: deterministic play was briefly worse
mid-training than at 600k, while sampled play stayed at 35. Worth remembering that a
single deterministic score is a noisy way to judge a checkpoint.

### Start (1, 2) is a different problem

It fails at every checkpoint, in both modes, so it is not caused by entropy collapse.
Running 200 sampled episodes from that start:

| model | coverage distribution | unvisited square |
|---|---|---|
| 1.2M checkpoint | 35 in 177 runs, lower in 23 | (0, 0) in all 177 |
| 3.0M final | 35 in 200 of 200 runs | (0, 0) in all 200 |

The policy always strands exactly one square, and it is always the corner (0, 0).

The reason is a property of the board, and it makes (1, 2) the hardest start of all 36.
Corner (0, 0) has only two knight-neighbours, (1, 2) and (2, 1). When the tour starts on
(1, 2), the corner can therefore only be taken in one of two ways: immediately as the
second square, or right at the end as the last square. Anything else strands it. Both
options were verified to lead to a full tour:

- taking (0, 0) as the 2nd square: tour found, ends at (2, 2)
- refusing (0, 0) as the 2nd square: tour found, ends at (0, 0)

So the board is fine and the policy is one decision short. It commits to a normal
opening move, and by the time it would need to come back the corner is unreachable.

Supporting evidence that this is learned rather than inherent: start (2, 1) is the mirror
image of (1, 2) under the board's diagonal symmetry and carries exactly the same
constraint, yet the policy solves it. There are 8 such corner-neighbour starts in total
(two per corner: (1,2), (2,1), (2,4), (1,3), (3,1), (4,2), (4,3), (3,4)) and the policy
solves 7 of them.

### What would fix it

- Keep exploration alive longer: schedule `ent_coef` down slowly rather than letting
  entropy collapse by 1.6M steps, so retries stay useful at evaluation time.
- Oversample hard starts. The 8 corner-neighbour squares are the difficult class, and
  uniform random starts give each only 1/36 of the training signal.
- Train longer or widen the network, the plain "more capacity, more steps" route.
- One-step lookahead or MCTS at inference would fix it outright, and it is worth noting
  this adds search, not heuristic knowledge, so it does not leak Warnsdorff into the
  agent.

Deliberately not doing: rewarding low-degree squares. That is Warnsdorff's rule in the
reward function, which would defeat the point of the experiment.
