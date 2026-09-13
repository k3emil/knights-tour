"""Gymnasium environment for the Knight's Tour problem.

The agent starts on a random square and must visit every square of an N x N board
exactly once, moving only as a chess knight. No heuristic (Warnsdorff or otherwise)
is encoded here: the agent only ever sees the board state and receives reward for
covering new squares.

Observation (MLP variant): 3 planes of N x N, flattened into a single vector.
    plane 0 - visited squares (1.0 = already visited)
    plane 1 - current knight position (1.0 on exactly one square)
    plane 2 - squares reachable by a legal move right now

Action: one of the 8 knight moves, indexed by KNIGHT_MOVES below.

Reward: +1.0 for every newly visited square, plus COMPLETION_BONUS when the whole
board is covered. Illegal actions are prevented by action masking, but are also
handled defensively with a small penalty in case masking is switched off.
"""

# ---------------------------------------------------------------------------------
# Imports. gymnasium supplies the Env base class and the "spaces" types that describe
# what an observation and an action look like; numpy holds the board and the planes.
# ---------------------------------------------------------------------------------
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------------
# The knight's 8 possible jumps, as (row delta, column delta) pairs. The ORDER matters
# and is fixed forever: action 0 always means (-2, -1), action 7 always means (2, 1).
# The network learns which index to choose, so renumbering these would invalidate a
# trained model.
# ---------------------------------------------------------------------------------
# (row delta, column delta) for the 8 knight moves
KNIGHT_MOVES = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
]

# ---------------------------------------------------------------------------------
# Reward constants.
#   COMPLETION_BONUS      - paid once, on top of the per-square rewards, when the board
#                           is fully covered. It makes "finish the tour" clearly better
#                           than "cover almost everything and get stuck".
#   ILLEGAL_ACTION_PENALTY - a safety net only. With action masking on, an illegal move
#                           can never be selected, so this is never actually paid.
# ---------------------------------------------------------------------------------
COMPLETION_BONUS = 10.0
ILLEGAL_ACTION_PENALTY = -1.0


class KnightsTourEnv(gym.Env):
    """Knight's Tour as a single-agent RL problem."""

    # Declares the supported render modes. "ansi" means render() returns text.
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, board_size: int = 6, flatten: bool = True):
        # gym.Env's constructor. Always call it first.
        super().__init__()

        # Board geometry. n_squares is both the number of cells and the number of
        # steps a complete tour takes (the start square counts as step 1).
        self.board_size = board_size
        self.flatten = flatten
        self.n_squares = board_size * board_size

        # --- Tell SB3 the shape of the problem -------------------------------------
        # 8 discrete actions: one per knight move.
        self.action_space = spaces.Discrete(len(KNIGHT_MOVES))

        # Observation shape depends on the network in use:
        #   flatten=True  -> one long vector (3 * N * N), what MlpPolicy expects
        #   flatten=False -> a 3-channel image (3, N, N), what a CnnPolicy expects
        # This single flag is the only difference between the MLP and CNN variants.
        if flatten:
            obs_shape: tuple[int, ...] = (3 * self.n_squares,)
        else:
            obs_shape = (3, board_size, board_size)
        # Every value in every plane is either 0.0 or 1.0, hence low=0, high=1.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)

        # --- Internal state --------------------------------------------------------
        # board stores the VISIT ORDER, not just occupancy: board[i][j] == 7 means the
        # knight arrived at (i, j) on move 7. This is what lets us print a readable
        # tour at the end, in the same format as the other two solvers.
        # step order of the tour: board[i][j] = 1..n_squares, 0 = unvisited
        self.board = np.zeros((board_size, board_size), dtype=np.int32)
        self.position: tuple[int, int] = (0, 0)
        self.start: tuple[int, int] = (0, 0)
        self.visited_count = 0

    # ------------------------------------------------------------------ helpers

    def _in_bounds(self, row: int, col: int) -> bool:
        # Is this square on the board at all?
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def _is_legal(self, row: int, col: int) -> bool:
        # A square may be jumped to only if it exists and has not been visited yet.
        # These are the rules of the puzzle, nothing strategic.
        return self._in_bounds(row, col) and self.board[row, col] == 0

    def action_masks(self) -> np.ndarray:
        """True for each knight move that stays on the board and lands on a new square."""
        # MaskablePPO calls this to learn which of the 8 actions are even possible, so
        # it never wastes training on off-board or repeat moves. Note this encodes only
        # legality - it says nothing about which legal move is a GOOD move, so no
        # heuristic leaks into the agent here.
        row, col = self.position
        return np.array(
            [self._is_legal(row + dr, col + dc) for dr, dc in KNIGHT_MOVES],
            dtype=bool,
        )

    def _observation(self) -> np.ndarray:
        # Build the three planes the network sees.

        # Plane 0: where the knight has already been (1.0 = visited).
        visited = (self.board > 0).astype(np.float32)

        # Plane 1: where the knight is standing right now (a single 1.0).
        current = np.zeros_like(visited)
        current[self.position] = 1.0

        # Plane 2: the squares it could legally jump to from here. Strictly speaking
        # this is derivable from the other two planes, but handing it over directly
        # saves the network from having to learn the knight's move pattern first.
        reachable = np.zeros_like(visited)
        row, col = self.position
        for dr, dc in KNIGHT_MOVES:
            if self._is_legal(row + dr, col + dc):
                reachable[row + dr, col + dc] = 1.0

        # Stack into (3, N, N), then flatten to a 3*N*N vector for the MLP if asked.
        obs = np.stack([visited, current, reachable], axis=0)
        return obs.reshape(-1) if self.flatten else obs

    def _info(self) -> dict[str, Any]:
        # Extra per-step diagnostics. These are not used for learning; the training
        # callback and the evaluator read them to report coverage and success rate.
        return {
            "squares": int(self.visited_count),
            "is_success": bool(self.visited_count == self.n_squares),
            "start": self.start,
        }

    # --------------------------------------------------------------- gym API

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Starts a new episode: clears the board and places the knight.
        # Seeds self.np_random so episodes are reproducible.
        super().reset(seed=seed)

        # Two ways to choose the starting square:
        #   options={"start": (r, c)} - a fixed square, used by evaluate.py to test
        #                               every start systematically
        #   nothing passed            - a random square, used during training so the
        #                               agent has to generalise instead of memorising
        #                               one single tour
        start = (options or {}).get("start")
        if start is None:
            flat = int(self.np_random.integers(self.n_squares))
            start = (flat // self.board_size, flat % self.board_size)

        # Wipe the board and mark the starting square as visit number 1.
        self.board[:] = 0
        self.position = (int(start[0]), int(start[1]))
        self.start = self.position
        self.board[self.position] = 1
        self.visited_count = 1

        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Applies one knight move and reports the outcome.
        # Translate the action index (0-7) into an actual target square.
        dr, dc = KNIGHT_MOVES[int(action)]
        row, col = self.position[0] + dr, self.position[1] + dc

        # Defensive branch: unreachable while action masking is enabled.
        if not self._is_legal(row, col):
            # Only reachable when action masking is disabled.
            return self._observation(), ILLEGAL_ACTION_PENALTY, True, False, self._info()

        # Perform the jump and record the visit order.
        self.position = (row, col)
        self.visited_count += 1
        self.board[row, col] = self.visited_count

        # Reward: one point per new square, plus the bonus if that square completed
        # the board. Paying per square is what makes this learnable - a completion-only
        # reward would be almost never earned by early random play.
        reward = 1.0
        solved = self.visited_count == self.n_squares
        if solved:
            reward += COMPLETION_BONUS

        # The episode is over either in triumph (board full) or in a dead end (no legal
        # move left). Both cases just stop; there is no backtracking in RL.
        # Episode ends on a full board or when the knight is trapped.
        terminated = solved or not self.action_masks().any()
        return self._observation(), reward, terminated, False, self._info()

    def render(self) -> str:
        # The Gymnasium hook, kept as a thin wrapper so render_board() can also be
        # called directly from the evaluator.
        return self.render_board()

    def render_board(self) -> str:
        """Zero-padded board, matching the output format of the other two solvers."""
        # width is 2 for a 6x6 board (36) and would be 3 for boards past 99 squares,
        # so columns stay aligned at any size.
        width = len(str(self.n_squares))
        rows = [
            " ".join(str(value).zfill(width) for value in row)
            for row in self.board.tolist()
        ]
        return "\n".join(rows)
