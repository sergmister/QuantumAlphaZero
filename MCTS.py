import copy
import math
import logging
import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, model, numMCTSSims=50, cpuct=1.0):
        self.game = game
        self.model = model
        self.numMCTSSims = numMCTSSims
        self.cpuct = cpuct
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        self.tree = {}

    def getActionProb(self, state, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.numMCTSSims):
            self.search(state)
            # print("Qsa")
            # for key, value in self.Qsa.items():
            #     print(key[0])
            #     print(key[1])
            #     print(value)

        s = str(state)
        counts = [
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
            for a in range(self.game.num_distinct_actions())
        ]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, state):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = str(state)

        if s not in self.Es:
            self.Es[s] = state.returns()[1 - state._cur_player]
        if state.is_terminal():
            # terminal node
            # print(s)
            # print(self.Es[s])
            # print(state._cur_player, state.returns())
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            # self.Ps[s], v = self.model.predict(
            #     np.array(state.observation_tensor(state._cur_player)).reshape(
            #         2, self.game.n, self.game.n
            #     )
            # )
            inarr = np.stack((state.board, np.full((self.game.n, self.game.n), state._get_turn(state.current_player()))))
            self.Ps[s], v = self.model.predict(inarr)
            v *= state._get_turn(state.current_player())
            valids = np.zeros(self.game.num_distinct_actions(), dtype=np.float32)
            valids[state.legal_actions()] = 1.0
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        num_valids = np.sum(valids)
        cur_best = -float("inf")
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.num_distinct_actions()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    # u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                    #     1 + self.Nsa[(s, a)]
                    # )
                    u = self.Qsa[(s, a)] + self.cpuct * (1 / num_valids) * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    # u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
                    u = self.cpuct * (1 / num_valids) * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = copy.deepcopy(state)
        next_s.apply_action(a)
        if s not in self.tree:
            self.tree[s] = []
        self.tree[s].append((str(next_s), a))

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
