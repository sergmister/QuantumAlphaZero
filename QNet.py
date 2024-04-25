import math
import numpy as np
from tqdm import tqdm

import pennylane as qml
from pennylane import numpy as p_np

from pennylane.templates.state_preparations import MottonenStatePreparation
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer, QNGOptimizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import AverageMeter


class QNetWrapper:
    def __init__(self, game, num_layers=8, use_gpu=True) -> None:
        self.board_x, self.board_y = game.n, game.n
        self.num_layers = num_layers
        self.n_qubits = math.ceil(math.log2(self.board_x * self.board_y + 1))
        self.use_gpu = use_gpu

        if self.use_gpu:
            dev = qml.device("lightning.gpu", wires=self.n_qubits)
            print("Using GPU with lightning.gpu")
        else:
            dev = qml.device("lightning.qubit", wires=self.n_qubits)
            print("Using CPU with lightning.qubit")

        # @qml.qnode(dev, diff_method="backprop")
        @qml.qnode(dev, diff_method="adjoint")
        def circuit(weights, inputs):
            """Quantum QVC Circuit"""

            # Input normalization
            inputs_1 = inputs.flatten() / p_np.sqrt(max(p_np.sum(inputs**2, axis=-1), 0.001))

            MottonenStatePreparation(inputs_1, wires=range(self.n_qubits))
            for i in range(len(weights)):
                StronglyEntanglingLayers(weights[i], wires=range(self.n_qubits))

            return qml.expval(qml.PauliZ(0))

        def cost(weights, X, Y):
            predictions = [circuit(weights, x) for x in X]
            return p_np.mean((Y - qml.math.stack(predictions)) ** 2)

        self.circuit = circuit
        self.cost = cost

        self.weights = p_np.random.uniform(
            low=-np.pi / 2,
            high=np.pi / 2,
            size=(self.num_layers, 1, self.n_qubits, 3),
            requires_grad=True,
        )
        self.total_params = np.prod(self.weights.shape)
        # self.opt = NesterovMomentumOptimizer(0.01)
        self.opt = AdamOptimizer(0.01)  # Seems to be faster than NesterovMomentumOptimizer
        # self.opt = QNGOptimizer() # Does not work directly
        self.epochs = 3
        # self.batch_size = 64
        self.batch_size = 256

    def train2(self, X, Y):
        for it in range(100):
            self.weights = self.opt.step(self.cost, self.weights, X=X, Y=Y)
            current_cost = self.cost(self.weights, X, Y)
            print(f"Cost: {current_cost}")

    def train(self, examples):
        # t = tqdm(range(self.epochs), desc="Training QNet")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}")
            batch_count = int(len(examples) / self.batch_size)

            v_losses = AverageMeter()
            t = tqdm(range(batch_count), desc="Training Net")
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.batch_size)
                boards, players, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                xs = p_np.zeros((len(boards), 2**self.n_qubits), dtype=np.float32)
                for i in range(len(boards)):
                    xs[i][: (self.board_x * self.board_y)] = boards[i].flatten()
                    xs[i][(self.board_x * self.board_y) :] = players[i]
                vs = p_np.array(vs).astype(np.float32)

                self.weights, _cost = self.opt.step_and_cost(self.cost, self.weights, X=xs, Y=vs)
                v_losses.update(_cost, len(xs))
                t.set_postfix(Loss_v=v_losses)

    def predict(self, board, player):
        x = p_np.zeros(2**self.n_qubits, dtype=np.float32)
        x[: (self.board_x * self.board_y)] = board.flatten()
        x[(self.board_x * self.board_y) :] = player
        return np.ones(self.board_x * self.board_y, dtype=np.float32), self.circuit(self.weights, x)
