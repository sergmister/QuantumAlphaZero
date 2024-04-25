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


class HybridQNetWrapper:
    def __init__(self, game, lr=0.005, use_gpu=True) -> None:
        self.board_x, self.board_y = game.n, game.n
        self.num_layers = 6
        self.n_qubits = 6
        self.lr = lr
        self.use_gpu = use_gpu
        self.cuda = False

        if self.use_gpu:
            dev = qml.device("lightning.gpu", wires=self.n_qubits)
            print("Using GPU with lightning.gpu")
        else:
            # dev = qml.device("lightning.qubit", wires=self.n_qubits)
            dev = qml.device("default.qubit", wires=self.n_qubits)
            print("Using CPU with lightning.qubit")

        @qml.qnode(dev, diff_method="backprop")
        # @qml.qnode(dev, diff_method="adjoint")
        def circuit(inputs, weights):
            """Quantum QVC Circuit"""

            # # Input normalization
            # inputs_1 = inputs / torch.sqrt(
            #     torch.max(
            #         torch.sum(inputs**2, axis=-1, dtype=torch.float32),
            #         torch.tensor(0.001, dtype=torch.float32),
            #     ).reshape(-1, 1)
            # )
            # try:
            #     MottonenStatePreparation(inputs_1, wires=range(self.n_qubits))
            # except:
            #     print(
            #         inputs,
            #         inputs_1,
            #         torch.max(
            #             torch.sum(inputs**2, axis=-1, dtype=torch.float32),
            #             torch.tensor(0.001, dtype=torch.float32),
            #         ).reshape(-1, 1),
            #     )
            #     raise Exception
            # StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            inputs = inputs.reshape(-1, 3, self.n_qubits)
            # switch dimensions 0 and 1
            inputs = inputs.permute(1, 0, 2)
            qml.AngleEmbedding(inputs[0], wires=range(self.n_qubits), rotation="X")
            qml.AngleEmbedding(inputs[1], wires=range(self.n_qubits), rotation="Y")
            qml.AngleEmbedding(inputs[2], wires=range(self.n_qubits), rotation="Z")
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit
        # self.circuit = qml.transforms.broadcast_expand(circuit)

        weight_shapes = {"weights": (self.num_layers, self.n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(self.circuit, weight_shapes)

        # self.fc1 = nn.Linear(self.board_x * self.board_y + 1, 2**self.n_qubits)
        self.fc1 = nn.Linear(self.board_x * self.board_y + 1, 3 * self.n_qubits)
        self.fc2 = nn.Linear(self.n_qubits, 1)
        self.tanh = nn.Tanh()

        # reshape layer from [3 * n_qubits] to [3, n_qubits]
        self.model = nn.Sequential(self.fc1, self.qlayer, self.fc2, self.tanh)

        self.total_params = sum(p.numel() for p in self.model.parameters())

        self.opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-5)
        self.loss = nn.MSELoss()

        self.epochs = 10
        self.batch_size = 256

    def train(self, examples):
        # t = tqdm(range(self.epochs), desc="Training QNet")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}")
            batch_count = int(len(examples) / self.batch_size)

            # print(self.fc1.weight)

            v_losses = AverageMeter()
            t = tqdm(range(batch_count), desc="Training Net")
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.batch_size)
                boards, players, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                xs = np.zeros((len(boards), self.board_x * self.board_y + 1), dtype=np.float32)
                for i in range(len(boards)):
                    xs[i][: (self.board_x * self.board_y)] = boards[i].flatten()
                    xs[i][(self.board_x * self.board_y) :] = players[i]
                xs = torch.FloatTensor(xs)

                vs = torch.FloatTensor(p_np.array(vs).astype(np.float32)).reshape(-1, 1)

                if self.cuda:
                    xs = xs.contiguous().cuda()
                    vs = vs.contiguous().cuda()

                self.opt.zero_grad()
                loss = self.loss(self.model(xs), vs)
                loss.backward()
                self.opt.step()

                v_losses.update(loss, len(xs))
                t.set_postfix(Loss_v=v_losses)

    def predict(self, board, player):

        # print the weights of self.fc1
        # print(self.fc1.weight)

        # preparing input
        x = torch.FloatTensor(np.concatenate((board.flatten(), [player]), dtype=np.float32))
        if self.cuda:
            x = x.contiguous().cuda()
        x = x.view(1, self.board_x * self.board_y + 1)

        self.model.eval()
        with torch.no_grad():
            v = self.model(x)

        return np.ones(self.board_x * self.board_y, dtype=np.float32), v.data.cpu().numpy()[0]
