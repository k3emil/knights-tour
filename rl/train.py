"""Train a MaskablePPO + MLP policy on the Knight's Tour environment.

The agent is given no heuristic: the only guidance is the reward (+1 per new square,
bonus on completion) and the action mask (which merely enforces the rules of chess).

Usage:
    python train.py --steps 3000000 --board-size 6 --n-envs 6
"""

# ---------------------------------------------------------------------------------
# Imports.
#   MaskablePPO      - PPO with action masking, from sb3-contrib
#   BaseCallback     - hook that runs during training; used here for custom metrics
#   CheckpointCallback - saves the model periodically so a crash is not fatal
#   make_vec_env     - builds several copies of the env at once
#   SubprocVecEnv    - runs those copies in separate OS processes (real parallelism)
# ---------------------------------------------------------------------------------
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from knights_tour_env import KnightsTourEnv

# Directory of this file, so logs and models land next to the script regardless of
# where it is launched from.
HERE = Path(__file__).parent


class TourStatsCallback(BaseCallback):
    """Logs rolling tour success rate and mean squares covered to TensorBoard."""

    # SB3 logs generic RL metrics (reward, losses) by itself. This callback adds the
    # two numbers that actually matter for this problem: what fraction of episodes
    # completed a tour, and how many squares were covered on average.

    def __init__(self, window: int = 200):
        super().__init__()
        # Metrics are averaged over the last `window` episodes rather than all of
        # training, so the curves show current behaviour instead of a lifetime average.
        self.window = window
        self.successes: list[float] = []
        self.squares: list[float] = []
        self.best_success_rate = 0.0

    def _on_step(self) -> bool:
        # Called after every environment step. self.locals is SB3's internal state;
        # "infos" holds one info dict per parallel environment.
        for info in self.locals.get("infos", []):
            # The Monitor wrapper adds an "episode" key only on the final step of an
            # episode, so this filter keeps one record per finished episode.
            episode = info.get("episode")
            if episode is None:
                continue
            self.successes.append(float(info.get("is_success", 0.0)))
            self.squares.append(float(info.get("squares", 0.0)))

        # Keep only the most recent `window` entries (a simple rolling buffer).
        self.successes = self.successes[-self.window :]
        self.squares = self.squares[-self.window :]

        # Write the metrics out for TensorBoard and the console table.
        if self.successes:
            success_rate = float(np.mean(self.successes))
            self.best_success_rate = max(self.best_success_rate, success_rate)
            self.logger.record("tour/success_rate", success_rate)
            self.logger.record("tour/squares_mean", float(np.mean(self.squares)))
            self.logger.record("tour/squares_max", float(np.max(self.squares)))
        # Returning True means "keep training". Returning False would stop early.
        return True


def build_env(board_size: int, n_envs: int, seed: int):
    # Creates n_envs independent boards that step in parallel, which is how PPO
    # collects experience quickly. Each gets a different seed, so they explore
    # different starting squares and different tours.
    return make_vec_env(
        KnightsTourEnv,
        n_envs=n_envs,
        seed=seed,
        # flatten=True gives the 108-value vector that MlpPolicy expects.
        env_kwargs={"board_size": board_size, "flatten": True},
        # SubprocVecEnv = one OS process per env, so multiple CPU cores are used.
        vec_env_cls=SubprocVecEnv,
        # Asks the Monitor wrapper to carry these two custom fields through to the
        # callback above.
        monitor_kwargs={"info_keywords": ("is_success", "squares")},
    )


def main() -> None:
    # --- Command-line options ------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--board-size", type=int, default=6)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-out", type=str, default=str(HERE / "model.zip"))
    args = parser.parse_args()

    env = build_env(args.board_size, args.n_envs, args.seed)

    # --- The agent -----------------------------------------------------------------
    # Hyperparameters, in plain terms:
    #   learning_rate 3e-4 - standard PPO step size
    #   n_steps  512  - each env collects 512 steps before an update (512 * n_envs total)
    #   batch_size 512 - minibatch size used inside each update
    #   n_epochs  10  - how many passes over that collected batch per update
    #   gamma  0.995  - discount factor; high because reward accrues over ~36 steps and
    #                   the completion bonus arrives right at the end
    #   gae_lambda / clip_range - standard PPO advantage smoothing and update clipping
    #   ent_coef 0.01 - entropy bonus, encourages exploration early on
    #   net_arch [128, 128] - two hidden layers of 128 units; this IS the whole network
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        tensorboard_log=str(HERE / "tb"),
        seed=args.seed,
        verbose=1,
    )

    # --- Callbacks -----------------------------------------------------------------
    stats = TourStatsCallback()
    # save_freq is counted per environment, so dividing by n_envs makes the interval
    # roughly every 100k total steps regardless of how many envs are running.
    checkpoints = CheckpointCallback(
        save_freq=100_000 // args.n_envs,
        save_path=str(HERE / "checkpoints"),
        name_prefix="ktour_mlp",
    )

    # --- Train ---------------------------------------------------------------------
    # This single call runs the whole RL loop: collect rollouts, compute advantages,
    # update the network, repeat until total_timesteps is reached.
    started = datetime.now()
    model.learn(total_timesteps=args.steps, callback=[stats, checkpoints], progress_bar=False)
    elapsed = datetime.now() - started

    # Persist the finished weights, then shut down the worker processes.
    model.save(args.model_out)
    env.close()

    # --- Summary -------------------------------------------------------------------
    # throughput is the useful number for planning bigger runs: multiply the steps you
    # want by this rate to estimate wall clock.
    steps_per_second = args.steps / max(elapsed.total_seconds(), 1e-9)
    print(f"\nTRAINING DONE  board={args.board_size}x{args.board_size}  policy=MlpPolicy")
    print(f"  timesteps            : {args.steps:,}")
    print(f"  wall clock           : {elapsed}")
    print(f"  throughput           : {steps_per_second:,.0f} steps/s")
    print(f"  best rolling success : {stats.best_success_rate:.3f}")
    print(f"  model saved to       : {args.model_out}")


# SubprocVecEnv spawns child processes that re-import this file, so the guard below is
# required on macOS and Windows - without it, each child would start its own training.
if __name__ == "__main__":
    main()
