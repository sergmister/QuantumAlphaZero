import numpy as np
from tqdm import tqdm

import pennylane as qml
from pennylane import numpy as p_np

from pennylane.templates.state_preparations import MottonenStatePreparation
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import AverageMeter

n_qubits = 5
# num_layers = 8
num_layers = 4

# dev = qml.device("default.qubit", wires=5)
dev = qml.device("lightning.qubit", wires=5)


# @qml.qnode(dev, diff_method="backprop")
@qml.qnode(dev, diff_method="adjoint")
def circuit(weights, inputs):
    """Quantum QVC Circuit"""

    inputs = inputs.flatten()

    # Input normalization
    inputs_1 = inputs / p_np.sqrt(max(p_np.sum(inputs**2, axis=-1), 0.001))

    MottonenStatePreparation(inputs_1, wires=range(n_qubits))

    for i, W in enumerate(weights):
        # Data re-uploading technique
        # if i % 4 == 0:
        #     MottonenStatePreparation(inputs_1, wires=range(n_qubits))

        # Neural network layer
        StronglyEntanglingLayers(weights[i], wires=range(n_qubits))

    # Measurement return
    # return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return qml.expval(qml.PauliZ(0))


def cost(weights, X, Y):
    predictions = [circuit(weights, x) for x in X]
    return p_np.mean((Y - qml.math.stack(predictions)) ** 2)


class QNetWrapper:
    def __init__(self, game) -> None:
        self.board_x, self.board_y = game.n, game.n

        self.weights = p_np.random.uniform(
            low=-np.pi / 2, high=np.pi / 2, size=(num_layers, 1, n_qubits, 3), requires_grad=True
        )
        self.circuit = circuit
        self.opt = NesterovMomentumOptimizer(0.01)
        self.epochs = 5
        self.batch_size = 64

    def train2(self, X, Y):
        for it in range(100):
            self.weights = self.opt.step(cost, self.weights, X=X, Y=Y)
            current_cost = cost(self.weights, X, Y)
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
                xs = p_np.zeros((len(boards), 2, self.board_x, self.board_y), dtype=np.float32)
                for i in range(len(boards)):
                    xs[i][0] = boards[i]
                    xs[i][1] = players[i]
                vs = p_np.array(vs).astype(np.float32)

                self.weights, _cost = self.opt.step_and_cost(cost, self.weights, X=xs, Y=vs)
                v_losses.update(_cost, len(xs))
                t.set_postfix(Loss_v=v_losses)

    def predict(self, x):
        return np.ones(self.board_x * self.board_y, dtype=np.float32), self.circuit(self.weights, x)
