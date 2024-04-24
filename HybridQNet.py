import numpy as np
from tqdm import tqdm
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import transpile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class HybridQNetWrapper:
    def __init__(self, game) -> None:
        self.board_x, self.board_y = game.n, game.n

        self.qc = QuantumCircuit(10)
        self.feature_map = ZZFeatureMap(2 * self.board_x * self.board_y, reps=2)
        self.ansatz = RealAmplitudes(10)
        return
        self.qc.compose(feature_map, inplace=True)
        self.qc.compose(ansatz, inplace=True)

    def train(self, examples):
        t = tqdm(range(self.epochs), desc="Training QNet")
        for epoch in t:
            self.nnet.train()

            batch_count = int(len(examples) / self.batch_size)

            # t = tqdm(range(batch_count), desc="Training Net")
            for _ in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=self.batch_size)
                boards, players, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                xs = np.zeros((len(boards), 2, self.board_x, self.board_y), dtype=np.float32)
                for i in range(len(boards)):
                    xs[i][0] = boards[i]
                    xs[i][1] = players[i]
