import os
import sys
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import AverageMeter


class OthelloNNet(nn.Module):
    def __init__(self, game, num_channels):
        # game params
        self.board_x, self.board_y = game.n, game.n
        self.action_size = game.num_distinct_actions()
        self.num_channels = num_channels
        self.dropout = 0.3

        super(OthelloNNet, self).__init__()

        self.conv1 = nn.Conv2d(1, self.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        self.bn3 = nn.BatchNorm2d(self.num_channels)
        self.bn4 = nn.BatchNorm2d(self.num_channels)

        self.fc1 = nn.Linear(self.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)  # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))  # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))  # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.num_channels * (self.board_x - 4) * (self.board_y - 4))

        s = F.dropout(
            F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training
        )  # batch_size x 1024
        s = F.dropout(
            F.relu(self.fc_bn2(self.fc2(s))), p=self.dropout, training=self.training
        )  # batch_size x 512

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


class OthelloNNet2(nn.Module):
    def __init__(self, game):
        # game params
        self.board_x, self.board_y = game.n, game.n
        self.action_size = game.num_distinct_actions()
        self.dropout = 0.2

        super(OthelloNNet2, self).__init__()

        self.fc1 = nn.Linear(2 * self.board_x * self.board_y, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)

        self.pi_out = nn.Linear(1024, self.action_size)
        self.v_out = nn.Linear(1024, 1)

    def forward(self, x):
        # concatenate the board and player
        # board = board.view(-1, self.board_x * self.board_y)
        # player = player.view(-1, self.board_x * self.board_y)
        # x = torch.cat((board, player), 1)
        x = x.view(-1, 2 * self.board_x * self.board_y)
        x = F.dropout(F.leaky_relu(self.fc1(x)), p=self.dropout, training=self.training)
        x = F.dropout(F.leaky_relu(self.fc2(x)), p=self.dropout, training=self.training)
        x = F.dropout(F.leaky_relu(self.fc3(x)), p=self.dropout, training=self.training)

        pi = self.pi_out(x)
        v = self.v_out(x)

        return F.log_softmax(pi, dim=1), torch.tanh(v)


class OthelloNNet3(nn.Module):
    def __init__(self, game, hidden_size):
        # game params
        self.board_x, self.board_y = game.n, game.n

        super(OthelloNNet3, self).__init__()

        self.fc1 = nn.Linear(self.board_x * self.board_y + 1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.v_out = nn.Linear(hidden_size, 1)

        self.total_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        x = x.view(-1, self.board_x * self.board_y + 1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        v = self.v_out(x)
        return torch.tanh(v)


class NNetWrapper:
    def __init__(self, game, hidden_size=8, lr=0.001):
        # self.nnet = OthelloNNet(game, 128)
        # self.nnet = OthelloNNet2(game)
        self.nnet = OthelloNNet3(game, hidden_size)
        self.board_x, self.board_y = game.n, game.n
        self.action_size = game.num_distinct_actions()
        self.epochs = 10
        self.batch_size = 256
        self.lr = lr
        self.cuda = torch.cuda.is_available()

        if self.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.lr, weight_decay=1e-4)

        # t = tqdm(range(self.epochs), desc="Training NNet")
        # pi_losses = AverageMeter()
        v_losses = AverageMeter()
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}")
            self.nnet.train()

            batch_count = int(len(examples) / self.batch_size)

            shuffled_indices = np.random.permutation(len(examples))

            t = tqdm(range(batch_count), desc="Training Net")
            for batch in t:
                # sample_ids = np.random.randint(len(examples), size=self.batch_size)
                # boards, players, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # use shuffled indices
                start_idx = batch * self.batch_size
                end_idx = (batch + 1) * self.batch_size
                if end_idx > len(examples):
                    end_idx = len(examples)
                indices = shuffled_indices[start_idx:end_idx]
                boards, players, pis, vs = list(zip(*[examples[i] for i in indices]))

                xs = np.zeros((len(boards), self.board_x * self.board_y + 1), dtype=np.float32)
                for i in range(len(boards)):
                    xs[i][: (self.board_x * self.board_y)] = boards[i].flatten()
                    xs[i][(self.board_x * self.board_y) :] = players[i]
                # boards = torch.FloatTensor(np.array(boards).astype(np.float32))
                xs = torch.FloatTensor(xs)
                # target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))

                # predict
                if self.cuda:
                    # boards = boards.contiguous().cuda()
                    xs = xs.contiguous().cuda()
                    # target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                # compute output
                # out_pi, out_v = self.nnet(xs)
                out_v = self.nnet(xs)
                # l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                # total_loss = l_pi + l_v
                total_loss = l_v

                # record loss
                # pi_losses.update(l_pi.item(), xs.size(0))
                v_losses.update(l_v.item(), xs.size(0))
                # t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)
                t.set_postfix(Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board, player):
        # preparing input
        x = torch.FloatTensor(np.concatenate((board.flatten(), [player]), dtype=np.float32))
        if self.cuda:
            x = x.contiguous().cuda()
        x = x.view(1, self.board_x * self.board_y + 1)
        self.nnet.eval()
        with torch.no_grad():
            # pi, v = self.nnet(x)
            v = self.nnet(x)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        # return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
        return np.ones(self.board_x * self.board_y, dtype=np.float32), v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save(
            {
                "state_dict": self.nnet.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.cuda else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint["state_dict"])
