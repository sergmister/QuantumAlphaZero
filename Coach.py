import logging
import random
import os
import sys
import copy
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import multiprocessing

import numpy as np
from tqdm import tqdm
import pyspiel
from open_spiel.python.algorithms import mcts, random_agent, minimax
from othello_game import OthelloGame, OthelloState

from Arena import Arena
from MCTS import MCTS
from othello_game import get_othello_symmetries

log = logging.getLogger(__name__)

# pool = multiprocessing.Pool(16)


class Coach:
    def __init__(
        self,
        game,
        model,
        num_iters=20,
        num_eps=5,
        numMCTSSims=25,
        compare_with=["random", "mcts"],
        compare_games=50,
        max_history_len=8192,
    ):
        self.game = game
        self.model = model
        self.num_iters = num_iters
        self.num_eps = num_eps
        self.numMCTSSims = numMCTSSims
        self.compare_with = compare_with
        self.compare_games = compare_games
        self.trainExamplesHistory = deque([], maxlen=max_history_len)
        self.performanceHistory = []

    def self_play(self):
        # [(board, current_player, pi, v)]
        trainExamples = []
        mcts = MCTS(self.game, self.model, numMCTSSims=self.numMCTSSims)
        state = self.game.new_initial_state()
        it = 0
        while not state.is_terminal():
            it += 1
            pi = mcts.getActionProb(state, temp=1)
            sym = get_othello_symmetries(copy.copy(state.board), np.array(pi, dtype=np.float32))
            for b, p in sym:
                trainExamples.append(
                    [
                        b,
                        # np.full(
                        #     (self.game.n, self.game.n), state._get_turn(state.current_player())
                        # ),
                        state._get_turn(state.current_player()),
                        p,
                        None,
                    ]
                )
            # trainExamples.append([copy.copy(state.board), np.full((self.game.n, self.game.n), state._get_turn(state.current_player())), pi, None])

            action = np.random.choice(len(pi), p=pi)
            state.apply_action(action)

        v = state.returns()[0]
        for e in trainExamples:
            e[3] = v

        return trainExamples

    def watch_play(self):
        trainExamples = []
        evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1)
        agent = mcts.MCTSBot(self.game, uct_c=2, max_simulations=10, evaluator=evaluator)
        state = self.game.new_initial_state()
        it = 0
        while not state.is_terminal():
            it += 1
            pi = np.zeros(self.game.num_distinct_actions())
            pi[state.legal_actions()] = 1
            pi = pi / np.sum(pi)
            sym = get_othello_symmetries(copy.copy(state.board), np.array(pi, dtype=np.float32))
            for b, p in sym:
                trainExamples.append(
                    [
                        b,
                        np.full(
                            (self.game.n, self.game.n), state._get_turn(state.current_player())
                        ),
                        p,
                        None,
                    ]
                )
            action = agent.step(state)
            state.apply_action(action)

        v = state.returns()[0]
        for e in trainExamples:
            e[3] = v

        return trainExamples

    def learn(self):
        for i in range(0, self.num_iters + 1):
            print(f"Iter {i}")

            if i > 0:
                for _ in tqdm(range(self.num_eps), desc="Self Play"):
                    self.trainExamplesHistory.extend(self.self_play())

                if len(self.trainExamplesHistory) >= 16:
                    self.model.train(self.trainExamplesHistory)

            if i % 4 == 0:
                print("Testing...")

                def get_alpha_zero_player():
                    class AlphaZeroPlayer:
                        def __init__(self, game, model, numMCTSSims):
                            self.model = model
                            self.mcts = MCTS(game, self.model, numMCTSSims=numMCTSSims)

                        def step(self, state):
                            probs = self.mcts.getActionProb(state, temp=1)
                            action = np.argmax(probs)
                            return action

                    return AlphaZeroPlayer(self.game, self.model, self.numMCTSSims)

                def get_random_player():
                    class RandomPlayer:
                        def __init__(self, game):
                            self.game = game

                        def step(self, state):
                            legal_actions = state.legal_actions()
                            action = random.choice(legal_actions)
                            return action

                    return RandomPlayer(self.game)

                def get_mcts_player():
                    class MCTSPlayer:
                        def __init__(self, game, numMCTSSims):
                            self.game = game
                            evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1)
                            self.mcts = mcts.MCTSBot(
                                game, uct_c=2, max_simulations=numMCTSSims, evaluator=evaluator
                            )

                        def step(self, state):
                            action = self.mcts.step(state)
                            return action

                    return MCTSPlayer(self.game, self.numMCTSSims)

                def get_minimax_player():
                    class MinimaxPlayer:
                        def __init__(self, game):
                            self.game = game

                        def step(self, state):
                            value, action = minimax.alpha_beta_search(self.game, state)
                            return action

                    return MinimaxPlayer(self.game)

                results = {}
                if "random" in self.compare_with:
                    results["random"] = Arena(
                        self.game, get_alpha_zero_player, get_random_player
                    ).playGames(self.compare_games)
                if "mcts" in self.compare_with:
                    results["mcts"] = Arena(
                        self.game, get_alpha_zero_player, get_mcts_player
                    ).playGames(self.compare_games)
                if "minimax" in self.compare_with:
                    results["minimax"] = Arena(
                        self.game, get_alpha_zero_player, get_minimax_player
                    ).playGames(self.compare_games)

                print(results)
                self.performanceHistory.append((i, results))
