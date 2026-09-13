"""Random-legal-move baseline for the Knight's Tour environment.

Establishes the bar the trained policy has to beat, and doubles as a check that the
environment, the reward and the action mask are wired up correctly.

Usage:
    python baseline_random.py --episodes 10000 --board-size 6
"""

# ---------------------------------------------------------------------------------
# No RL library is needed here. This script only drives the environment with random
# legal moves, so it depends on nothing but numpy and the env itself.
# ---------------------------------------------------------------------------------
import argparse
from datetime import datetime

import numpy as np

from knights_tour_env import KnightsTourEnv


def run(episodes: int, board_size: int, seed: int) -> None:
    # One environment instance, reused for every episode (reset() clears the board).
    env = KnightsTourEnv(board_size=board_size)
    # A seeded generator, so the whole baseline is reproducible.
    rng = np.random.default_rng(seed)

    # coverage[i] = how many squares episode i managed to visit.
    coverage = np.zeros(episodes, dtype=np.int32)
    successes = 0
    start = datetime.now()

    for episode in range(episodes):
        # Fresh board with a random starting square (no "start" option passed).
        env.reset(seed=int(rng.integers(2**31 - 1)))
        terminated = False

        # Walk until the knight is trapped or the board is full.
        while not terminated:
            # This is the "policy": take the mask of legal moves, turn it into a list
            # of legal action indices, and pick one uniformly at random. No strategy.
            legal = np.flatnonzero(env.action_masks())
            action = int(rng.choice(legal))
            _, _, terminated, _, info = env.step(action)

        # Record how far this episode got. info comes from the env's _info().
        coverage[episode] = info["squares"]
        successes += int(info["is_success"])

    # Report: success rate is the headline number the trained model must beat, and
    # mean coverage shows how far random play typically gets before dead-ending.
    elapsed = datetime.now() - start
    print(f"RANDOM BASELINE  board={board_size}x{board_size}  episodes={episodes}")
    print(f"  full tours      : {successes} ({100.0 * successes / episodes:.3f}%)")
    print(f"  squares covered : mean {coverage.mean():.2f}  max {coverage.max()}  min {coverage.min()}")
    print(f"  elapsed         : {elapsed}")


def main() -> None:
    # Command-line arguments, so board size and episode count can be varied without
    # editing the file.
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--board-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.episodes, args.board_size, args.seed)


if __name__ == "__main__":
    main()
