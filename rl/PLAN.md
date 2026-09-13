# Knight's Tour with Reinforcement Learning — Plans

Third solution method for this repo, alongside backtracking and Warnsdorff's rule.
Goal: show that RL can learn to complete knight's tours **without** being given any
heuristic, AlphaZero-style (learn from reward only). Speed comparison is deferred.

Board choice: **6x6**, because tours exist from all 36 start squares (verified by
search), so the target success rate is a clean 100%.

Two network variants are planned. Everything except the network and the observation
shape is identical between them.

**Status: Plan A (6x6 + MLP) is the chosen path and is being implemented now.**
Plan B (CNN) is kept as a later comparison run. It reuses the same environment, reward,
masking and evaluation, so the two are measured on equal terms and the only difference
is the network.

---

## Plan A — 6x6, MLP, local only — SELECTED, IN PROGRESS

### 0. Setup
Python venv, install `gymnasium`, `stable-baselines3`, `sb3-contrib`, `torch`,
`tensorboard`. Pin versions in `requirements.txt`.

### 1. Environment (`knights_tour_env.py`)
Gymnasium env: 6x6 board, random start from all 36 squares each episode (all verified
solvable). Observation = the same 3 planes (visited squares, knight position,
legal-move mask) flattened into a 108-value vector. 8 discrete actions,
`action_masks()` for legality, optional fixed start via `reset(options=...)` for
evaluation. Reward +1 per new square, plus ~10 bonus at 36. Episode ends when no legal
move remains.

### 2. Baseline
Random legal-move agent, 10k episodes. Record tour success rate and mean squares
covered. Confirms the env works and sets the bar to beat.

### 3. Network
MaskablePPO from sb3-contrib with `"MlpPolicy"`. No custom extractor needed — SB3's
default two hidden layers of 64 units work on a flat vector, and `net_arch=[128, 128]`
is a reasonable first bump if learning stalls.

### 4. Train
6 parallel envs (6 physical cores on the i7-9750H), ~2-5M steps, TensorBoard logging of
success rate and mean coverage, periodic checkpoints. Expect faster than the CNN run,
so roughly 15-30 min; budget an evening for a few hyperparameter passes.

### 5. Evaluate (`evaluate.py`)
Run all 36 starts deterministically: success rate and mean squares covered. Save the
model and print one finished tour as a **6x6** grid (the printed grid always follows the
env's board size, never a fixed 8x8), reusing the zero-padded two-digit cell style of
the existing two scripts.

### 6. Repo and reporting
Keep all RL work inside the `rl/` folder and write a new RL-specific `rl/README.md`
rather than expanding the top-level README. That RL README documents the method, the
environment, the reward and how to run training and evaluation, and it holds the results
of **both** variants (MLP and CNN) side by side in one comparison table: success rate,
mean squares covered, training steps and training time. The top-level `README.md` is
**not** touched at all — no new sections, no links, no edits of any kind.

### Later
CNN variant for comparison, inference-speed benchmarking, larger boards (CNN only, via
a fixed padded canvas with an off-board channel), MCTS if the policy plateaus, and
porting the same script to SageMaker.

---

## Plan B — 6x6, CNN, local only — LATER (comparison run)

### 0. Setup
Python venv, install `gymnasium`, `stable-baselines3`, `sb3-contrib`, `torch`,
`tensorboard`. Pin versions in `requirements.txt`.

### 1. Environment (`knights_tour_env.py`)
Gymnasium env: 6x6 board, random start from all 36 squares each episode (all verified
solvable). Observation = 3 channels of 6x6: visited squares, knight position,
legal-move mask. 8 discrete actions, `action_masks()` for legality, optional fixed
start via `reset(options=...)` for evaluation. Reward +1 per new square, plus ~10 bonus
at 36. Episode ends when no legal move remains.

### 2. Baseline
Random legal-move agent, 10k episodes. Record tour success rate and mean squares
covered. Confirms the env works and sets the bar to beat.

### 3. Network
MaskablePPO from sb3-contrib with a custom small CNN extractor — SB3's default
`NatureCNN` needs inputs >= 36x36, so 6x6 needs your own. Two conv layers, 3x3 kernels,
padding 1 (5x5 receptive field, matching the knight's reach), then flatten to a small
dense layer.

### 4. Train
6 parallel envs (6 physical cores on the i7-9750H), ~2-5M steps, TensorBoard logging of
success rate and mean coverage, periodic checkpoints. Expect roughly 20-45 min per run;
budget an evening for a few hyperparameter passes.

### 5. Evaluate (`evaluate.py`)
Run all 36 starts deterministically: success rate and mean squares covered. Save the
model and print one finished tour as a **6x6** grid (the printed grid always follows the
env's board size, never a fixed 8x8), reusing the zero-padded two-digit cell style of
the existing two scripts.

### 6. Repo and reporting
Keep all RL work inside the `rl/` folder and write a new RL-specific `rl/README.md`
rather than expanding the top-level README. That RL README documents the method, the
environment, the reward and how to run training and evaluation, and it holds the results
of **both** variants (MLP and CNN) side by side in one comparison table: success rate,
mean squares covered, training steps and training time. The top-level `README.md` is
**not** touched at all — no new sections, no links, no edits of any kind.

### Later
MLP variant for comparison, inference-speed benchmarking, larger boards via a fixed
padded canvas with an off-board channel, MCTS if the policy plateaus, and porting the
same script to SageMaker.

---

## MLP vs CNN — what differs

|                          | MLP (selected)                            | CNN (later)                                        |
|--------------------------|--------------------------------|--------------------------------------------|
| Observation              | 108-value flat vector          | 3 x 6 x 6 grid                             |
| Policy                   | `"MlpPolicy"` (default 64x64)  | custom small CNN extractor                 |
| Expected training speed  | faster                         | slower                                     |
| Expected inference speed | faster                         | slower                                     |
| Scales to larger boards  | no (position-specific weights) | yes (shared weights, padded canvas)        |

Code difference is small: swap the policy string and flatten the observation.

---

## Board facts established before starting (verified by exhaustive / heuristic search)

| Board | Tours exist from | Notes |
|-------|------------------|-------|
| 3x3   | 0 of 9 starts    | centre square unreachable |
| 4x4   | 0 of 16 starts   | structurally impossible, not a parity issue |
| 5x5   | 13 of 25 starts  | parity: 13 majority-colour vs 12 minority; minority starts cap at 24 of 25 squares |
| 6x6   | 36 of 36 starts  | chosen for this project |
| 7x7   | 25 of 49 starts  | same odd-board parity restriction as 5x5 |
| 8x8   | 64 of 64 starts  | target for later scaling |

Also measured: on 5x5 the repo's Warnsdorff tie-break (first minimum in a fixed move
order) completes only 9 of the 13 solvable starts, dead-ending at 17 or 19 squares on
the other 4. So plain Warnsdorff is not universal at every board size.

## Evaluation rules
- Success rate is measured over start squares from which a tour actually exists.
- No Warnsdorff knowledge in the reward, the observation, or the move ordering.
