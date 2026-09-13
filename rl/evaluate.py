"""Evaluate a trained Knight's Tour policy from every start square.

Reports the tour success rate and mean coverage, and prints one finished tour in the
same zero-padded format used by the backtracking and Warnsdorff scripts.

Usage:
    python evaluate.py --model model.zip --board-size 6
    python evaluate.py --model model.zip --attempts 10   # sampled retries per start
"""

# ---------------------------------------------------------------------------------
# Only MaskablePPO is needed at inference time, to load the saved weights. No training
# machinery, no parallel environments.
# ---------------------------------------------------------------------------------
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from knights_tour_env import KnightsTourEnv

HERE = Path(__file__).parent


def play(model: MaskablePPO, env: KnightsTourEnv, start, deterministic: bool):
    # Plays one complete episode from a given start square.
    # This is the inference loop, and it shows exactly how a trained model gets used:
    # the network is asked for one move at a time and the environment applies it. The
    # loop lives here, in ordinary Python - the model itself only ever answers the
    # question "given this board, which of the 8 moves?".
    obs, _ = env.reset(options={"start": start})
    terminated = False
    info: dict = {}
    while not terminated:
        # deterministic=True takes the highest-scoring legal move (argmax).
        # deterministic=False samples from the probability distribution instead.
        # action_masks is passed so the model can only choose a legal move.
        action, _ = model.predict(
            obs, action_masks=env.action_masks(), deterministic=deterministic
        )
        obs, _, terminated, _, info = env.step(int(action))
    # info carries the final square count and whether the tour completed.
    return info["squares"], info["is_success"]


def main() -> None:
    # --- Options -------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(HERE / "model.zip"))
    parser.add_argument("--board-size", type=int, default=6)
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="1 = deterministic (argmax). >1 = sample this many tries per start, "
        "counting a start as solved if any try completes.",
    )
    args = parser.parse_args()

    # Load the trained weights. device="cpu" keeps it predictable for timing later.
    model = MaskablePPO.load(args.model, device="cpu")
    env = KnightsTourEnv(board_size=args.board_size)
    # A single attempt implies argmax; more than one implies sampling, since repeating
    # a deterministic run would give the identical result every time.
    deterministic = args.attempts == 1

    coverage: list[int] = []
    solved_starts: list[tuple[int, int]] = []
    example_board = None

    # --- Try every start square on the board ---------------------------------------
    started = datetime.now()
    for row in range(args.board_size):
        for col in range(args.board_size):
            best = 0
            for _ in range(args.attempts):
                squares, success = play(model, env, (row, col), deterministic)
                # Keep the furthest this start ever got, across attempts.
                best = max(best, squares)
                if success:
                    solved_starts.append((row, col))
                    # Remember the first complete tour found, to print as an example.
                    if example_board is None:
                        example_board = env.render_board()
                    # No need for more attempts once this start is solved.
                    break
            coverage.append(best)
    elapsed = datetime.now() - started

    # --- Report --------------------------------------------------------------------
    # "starts solved" is the headline metric: on a 6x6 board a tour exists from all 36
    # squares, so anything below 36 is a gap in the policy rather than an impossible
    # board.
    n_starts = args.board_size**2
    n_solved = len(solved_starts)
    print(f"EVALUATION  board={args.board_size}x{args.board_size}  model={args.model}")
    print(f"  mode            : {'deterministic' if deterministic else f'{args.attempts} sampled attempts'}")
    print(f"  starts solved   : {n_solved} / {n_starts} ({100.0 * n_solved / n_starts:.1f}%)")
    print(f"  squares covered : mean {np.mean(coverage):.2f}  min {min(coverage)}  max {max(coverage)}")
    print(f"  elapsed         : {elapsed}")

    # List the specific squares that failed, which is what you need to debug or to
    # decide whether more training would help.
    if n_solved < n_starts:
        failed = [
            (r, c)
            for r in range(args.board_size)
            for c in range(args.board_size)
            if (r, c) not in solved_starts
        ]
        print(f"  unsolved starts : {failed}")

    # Print one finished tour in the same layout as the other two solvers, so the three
    # methods can be compared side by side by eye.
    if example_board is not None:
        print("\nEXAMPLE COMPLETE TOUR")
        print(example_board)
    else:
        print("\nNo complete tour found.")


if __name__ == "__main__":
    main()
