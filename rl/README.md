# Knight's Tour by Reinforcement Learning

A third solution method for this repo, next to backtracking and Warnsdorff's rule.

The question here is different from the first two scripts. Those implement an algorithm
someone already knew. This one asks whether an agent can **learn** to complete knight's
tours from nothing but reward, in the spirit of AlphaZero learning chess without opening
books. No heuristic is encoded anywhere: not in the reward, not in the observation, not
in the move ordering.

Board size is **6x6**, chosen because a tour exists from all 36 start squares, so the
target success rate is a clean 100%. (On 3x3 and 4x4 no tour exists at all; on 5x5 and
7x7 colour parity rules out every minority-colour start. See `PLAN.md`.)

Speed comparison against the other two methods is deliberately out of scope for now.

## Method

| piece | choice |
|---|---|
| algorithm | MaskablePPO (PPO with action masking), sb3-contrib |
| network | MLP, `net_arch=[128, 128]` |
| observation | 3 planes of 6x6 — visited squares, knight position, currently reachable squares — flattened to 108 values |
| actions | 8 knight moves |
| action mask | blocks off-board and already-visited targets, i.e. the rules of chess only |
| reward | +1 per newly visited square, +10 on covering all 36 |
| episode end | board complete, or knight trapped with no legal move |
| start square | uniformly random over all 36 squares each episode |

Random starts are on purpose. With a fixed start the agent can memorise one winning
sequence; varying the start forces it to learn something that generalises.

The action mask deserves a note, since it is the one place where knowledge is injected.
It only encodes legality (a knight cannot leave the board or revisit a square), which is
the definition of the puzzle rather than a strategy for solving it. All judgement about
*which* legal move is good remains learned.

## Files

| file | purpose |
|---|---|
| `knights_tour_env.py` | Gymnasium environment (observation, reward, action masking, board rendering) |
| `baseline_random.py` | random-legal-move agent, the bar the policy has to beat |
| `train.py` | MaskablePPO + MLP training, TensorBoard logging, checkpoints |
| `evaluate.py` | plays every start square, reports success rate and prints one finished tour |
| `PLAN.md` | full plan for both the MLP and CNN variants, plus the board facts behind the 6x6 choice |
| `RESULTS.md` | measured results with commands, settings and logs |

## Running it

```bash
cd rl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python baseline_random.py --episodes 10000 --board-size 6
python train.py --steps 3000000 --board-size 6 --n-envs 6
python evaluate.py --model model.zip --board-size 6
tensorboard --logdir tb          # optional, learning curves
```

`requirements.txt` is pinned for macOS on Intel, where PyTorch stopped publishing wheels
after 2.2.2; that pin in turn fixes the SB3 and NumPy versions.

## Results (6x6)

| method | starts solved | mean squares covered | notes |
|---|---|---|---|
| random legal moves | 0 / 36 | 20.10 | 10,000 episodes, never completed a tour |
| RL — MaskablePPO + MLP | 35 / 36 (97.2%) | 35.97 | 3M steps, 26 min on 6 CPU cores |
| RL — MaskablePPO + CNN | not run yet | — | planned variant, see `PLAN.md` |
| Warnsdorff's rule | 36 / 36 | 36 | hand-coded heuristic, for reference |

Training reached 31.2 mean squares by 132k steps and a 98% rolling success rate by about
1.15M steps. The one failing start, (1, 2), reaches 35 of 36 squares; a tour does exist
from there, so it is a policy gap rather than an impossible board.

Retrying with sampled actions instead of argmax did not help, because the policy's
entropy had collapsed by the end of training — sampling had become equivalent to argmax.
Decaying `ent_coef` more slowly would make retries a genuine second chance.

Full numbers, exact commands and log files are in `RESULTS.md`.

## What this shows

An agent with no knowledge of Warnsdorff's rule, learning only from "+1 per new square",
goes from 0/36 to 35/36 solved starts. That answers the original question: RL does apply
to the knight's tour. It does not beat the hand-coded heuristic on this board, which is
the expected and honest outcome — Warnsdorff is a very good rule for a problem this size.

## Next steps

- CNN variant, for comparison on success rate, training time and inference time
- Close the last start square (longer training, slower entropy decay, or a larger network)
- Inference-speed benchmark against backtracking and Warnsdorff
- Larger boards (CNN only, padding boards onto a fixed canvas with an off-board channel)
- MCTS on top of the policy, if plain policy improvements plateau
- Port the same training script to SageMaker as an AWS exercise
