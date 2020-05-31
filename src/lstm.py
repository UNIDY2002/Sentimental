import torch
from torch import nn
from torch.nn import functional as fun

from interface import Model
from util import WV_DIM, TARGET_DIM

TOTAL_EPOCH = 30
LEARNING_RATE = 1.3
GAMMA = 0.85
WORD_PER_SENTENCE = 240
BATCH_SIZE = 1
HIDDEN_DIM = 40
CHANNEL_CNT = 60
DROPOUT = 0.2


class LSTM(Model):
    def __init__(self):
        super().__init__(TOTAL_EPOCH, LEARNING_RATE, GAMMA, WORD_PER_SENTENCE, 'full', BATCH_SIZE, 'lstm')
        self.lstm = nn.LSTM(WV_DIM, HIDDEN_DIM, num_layers=2, bidirectional=True, dropout=DROPOUT)
        self.conv = nn.Conv2d(1, CHANNEL_CNT, (8, HIDDEN_DIM * 2))
        self.pooling = nn.MaxPool1d(WORD_PER_SENTENCE - 7)
        self.linear = nn.Linear(CHANNEL_CNT, TARGET_DIM)

    def forward(self, inputs, batch_size):
        inputs = inputs.view(batch_size, -1, WV_DIM)
        out, _ = self.lstm(inputs.permute([1, 0, 2]),
                           (torch.rand(4, batch_size, HIDDEN_DIM), torch.rand(4, batch_size, HIDDEN_DIM)))
        out = self.conv(out.permute([1, 0, 2]).reshape(batch_size, 1, -1, HIDDEN_DIM * 2))
        out = self.pooling(out.view(CHANNEL_CNT * batch_size, 1, -1)).view(batch_size, -1)
        return fun.softmax(torch.tanh(self.linear(out)), dim=1)


LSTM().run()
