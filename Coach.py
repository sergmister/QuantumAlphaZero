import logging
import random
import os
import sys
import copy
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS
from othello_game import get_othello_symmetries

log = logging.getLogger(__name__)


class Coach:
    def __init__(self, game, model):
        self.game = game
        self.model = model
        self.prev_model = self.model.__class__(self.game)
        self.num_iters = 10
        self.num_eps = 10
        self.numMCTSSims = 50
        self.trainExamplesHistory = deque([], maxlen=1000)

    def self_play(self):
        # [(board, current_player, pi, v)]
        trainExamples = []
        mcts = MCTS(self.game, self.model, numMCTSSims=self.numMCTSSims)
        state = self.game.new_initial_state()
        it = 0
        while not state.is_terminal():
            it = 0
            pi = mcts.getActionProb(state, temp=1)
            sym = get_othello_symmetries(copy.copy(state.board), np.array(pi, dtype=np.float32))
            for b, p in sym:
                trainExamples.append([b, state.current_player(), p, None])

            action = np.random.choice(len(pi), p=pi)
            state.apply_action(action)

        v = state.returns()[0]
        for e in trainExamples:
            e[3] = v

        return trainExamples

    def learn(self):
        for i in range(1, self.num_iters + 1):
            log.info(f"Iter {i}")

            for _ in tqdm(range(self.num_eps), desc="Self Play"):
                self.trainExamplesHistory.extend(self.self_play())

            if len(self.trainExamplesHistory) > 100:
                self.model.train(self.trainExamplesHistory)

            log.info("Pitting against previous version")

            def get_alpha_zero_player():
                class AlphaZeroPlayer:
                    def __init__(self, game, model, numMCTSSims=50):
                        self.model = model
                        self.mcts = MCTS(game, self.model, numMCTSSims=numMCTSSims)

                    def step(self, state):
                        probs = self.mcts.getActionProb(state, temp=1)
                        # action = np.random.choice(len(probs), p=probs)
                        action = np.argmax(probs)
                        return action

                return AlphaZeroPlayer(self.game, self.model, numMCTSSims=self.numMCTSSims)

            def get_random_player():
                class RandomPlayer:
                    def __init__(self, game):
                        self.game = game

                    def step(self, state):
                        legal_actions = state.legal_actions()
                        action = random.choice(legal_actions)
                        return action

                return RandomPlayer(self.game)

            arena = Arena(self.game, get_alpha_zero_player, get_random_player)

            oneWon, twoWon, draws = arena.playGames(10)
            print(f"1: {oneWon}, 2: {twoWon}, d: {draws}")
