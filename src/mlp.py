import torch
from torch import nn
from torch.nn import functional as fun

from interface import Model
from util import WV_DIM, TARGET_DIM

TOTAL_EPOCH = 60
LEARNING_RATE = 0.9
GAMMA = 0.95
DROPOUT = 0.0


class MLP(Model):
    def __init__(self):
        super().__init__(TOTAL_EPOCH, LEARNING_RATE, GAMMA, 0, 'average', 1, 'mlp')
        self.linear1 = nn.Linear(WV_DIM, 9 * TARGET_DIM)
        self.linear2 = nn.Linear(9 * TARGET_DIM, 3 * TARGET_DIM)
        self.linear3 = nn.Linear(3 * TARGET_DIM, TARGET_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, inputs, _):
        hidden1 = torch.tanh(self.dropout(self.linear1(inputs)))
        hidden2 = torch.tanh(self.dropout(self.linear2(hidden1)))
        out = torch.tanh(self.linear3(hidden2))
        return fun.softmax(out.view(1, -1), dim=1)


MLP().run()
