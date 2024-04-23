import numpy as np
from open_spiel.python.observation import IIGObserverForPublicInfoGame
import pyspiel

DIRS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
DEFAULT_N = 6

num_players = 2
game_type = pyspiel.GameType(
    short_name="othello_nxn",
    long_name="Python Othello nxn",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=num_players,
    min_num_players=num_players,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={},
)


class OthelloGame(pyspiel.Game):
    def __init__(self, params=dict()):
        self.n = params.get("n", DEFAULT_N)
        assert self.n % 2 == 0
        num_cells = self.n * self.n
        game_info = pyspiel.GameInfo(
            num_distinct_actions=num_cells,
            max_chance_outcomes=0,
            num_players=2,
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0.0,
            max_game_length=num_cells,
        )
        super().__init__(game_type, game_info, params)

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return OthelloState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        if (iig_obs_type is None) or (iig_obs_type.public_info and not iig_obs_type.perfect_recall):
            return BoardObserver(params)
        else:
            return IIGObserverForPublicInfoGame(iig_obs_type, params)

    def num_distinct_actions(self):
        """Returns the number of distinct actions available to an agent."""
        return self.n * self.n


class OthelloState(pyspiel.State):
    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self.n = game.n
        self._cur_player = 0
        self._player0_score = 0.0
        self._is_terminal = False
        # 1 for black, -1 for white, 0 for empty
        self.board = np.zeros((self.n, self.n), np.int8)
        self.board[self.n // 2 - 1][self.n // 2 - 1] = -1
        self.board[self.n // 2][self.n // 2] = -1
        self.board[self.n // 2 - 1][self.n // 2] = 1
        self.board[self.n // 2][self.n // 2 - 1] = 1

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every perfect-information sequential-move game.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def _get_turn(self, player=None):
        if player is not None:
            return 1 if player == 0 else -1
        else:
            return 1 if self._cur_player == 0 else -1

    def _get_action(self, x, y):
        return x * self.n + y

    def _get_coord(self, action):
        return (action // self.n, action % self.n)

    def _in_board(self, x, y):
        return 0 <= x < self.n and 0 <= y < self.n

    def _legal_action(self, action, player=None):
        turn = self._get_turn(player)
        x, y = self._get_coord(action)
        if self.board[x][y] != 0:
            return False
        for xdir, ydir in DIRS:
            px = x
            py = y
            onum = False
            while True:
                px += xdir
                py += ydir
                if self._in_board(px, py):
                    if self.board[px][py] == -turn:
                        onum = True
                    elif self.board[px][py] == turn:
                        if onum:
                            return True
                        else:
                            break
                    else:
                        break
                else:
                    break

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        return [a for a in range(self.n * self.n) if self._legal_action(a, player)]

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        turn = self._get_turn()
        x, y = self._get_coord(action)
        self.board[x][y] = turn
        for xdir, ydir in DIRS:
            cpos = []
            px = x
            py = y
            found = False
            while True:
                px += xdir
                py += ydir
                if self._in_board(px, py):
                    if self.board[px][py] == 0:
                        break
                    elif self.board[px][py] == -turn:
                        cpos.append((px, py))
                    elif self.board[px][py] == turn:
                        if len(cpos) == 0:
                            break
                        else:
                            found = True
                            break
                else:
                    break
            if found:
                for xpos, ypos in cpos:
                    self.board[xpos][ypos] = turn

        if len(self._legal_actions(1 - self._cur_player)) == 0:
            self._is_terminal = True
            diff = np.sum(self.board)
            if diff > 0:
                self._player0_score = 1.0
            elif diff < 0:
                self._player0_score = -1.0
            else:
                self._player0_score = 0.0
        else:
            self._cur_player = 1 - self._cur_player

    def _action_to_string(self, player, action):
        """Action -> string."""
        x, y = self._get_coord(action)
        return f"({x}, {y})"

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._is_terminal

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return [self._player0_score, -self._player0_score]

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return _board_to_string(self.board)


class BoardObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, params):
        """Initializes an empty observation tensor."""
        self.n = params.get("n", DEFAULT_N)
        # The observation should contain a 1-D tensor in `self.tensor` and a
        # dictionary of views onto the tensor, which may be of any shape.
        # Here the observation is indexed `(cell state, row, column)`.
        shape = (num_players, self.n, self.n)
        self.tensor = np.zeros(np.prod(shape), np.float32)
        self.dict = {"observation": np.reshape(self.tensor, shape)}

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        # del player
        # We update the observation via the shaped tensor since indexing is more
        # convenient than with the 1-D tensor. Both are views onto the same memory.
        obs = self.dict["observation"]
        obs[0, :, :] = state.board == state._get_turn(player)
        obs[1, :, :] = state.board == -state._get_turn(player)

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        del player
        return _board_to_string(state.board)


def _board_to_string(board):
    """Returns a string representation of the board."""
    return "\n".join(
        " ".join("X" if c == 1 else "O" if c == -1 else "." for c in row) for row in board
    )


def get_othello_symmetries(board, pi):
    """Returns symmetries of the state (mirror, rotation)."""
    symmetries = []
    for i in range(1, 4):
        # 90 degree rotations
        rotated_board = np.rot90(board, i)
        flipped_board = np.fliplr(rotated_board)
        rotated_pi = np.rot90(pi.reshape(6, 6), i).flatten()
        flipped_pi = np.fliplr(rotated_pi.reshape(6, 6)).flatten()
        symmetries.append((flipped_board, flipped_pi))
        symmetries.append((rotated_board, rotated_pi))
    return symmetries


# Register the game with the OpenSpiel library

pyspiel.register_game(game_type, OthelloGame)
