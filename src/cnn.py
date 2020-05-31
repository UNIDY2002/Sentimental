import torch
from torch import nn
from torch.nn import functional as fun

from interface import Model
from util import WV_DIM, TARGET_DIM

TOTAL_EPOCH = 60
LEARNING_RATE = 1.5
GAMMA = 0.97
WORD_PER_SENTENCE = 150
BATCH_SIZE = 120
CHANNEL_CNT = 12


class CNN(Model):
    def __init__(self):
        super().__init__(TOTAL_EPOCH, LEARNING_RATE, GAMMA, WORD_PER_SENTENCE, 'full', BATCH_SIZE, 'cnn')
        self.conv1 = nn.Conv2d(1, CHANNEL_CNT, (12, WV_DIM))
        self.pooling1 = nn.MaxPool1d(WORD_PER_SENTENCE - 11)
        self.conv2 = nn.Conv2d(1, CHANNEL_CNT, (8, WV_DIM))
        self.pooling2 = nn.MaxPool1d(WORD_PER_SENTENCE - 7)
        self.conv3 = nn.Conv2d(1, CHANNEL_CNT, (6, WV_DIM))
        self.pooling3 = nn.MaxPool1d(WORD_PER_SENTENCE - 5)
        self.linear = nn.Linear(CHANNEL_CNT * 3, TARGET_DIM)

    def forward(self, inputs, batch_size):
        c_layer1 = torch.tanh(self.conv1(inputs.view(batch_size, 1, -1, WV_DIM)))
        p_layer1 = self.pooling1(c_layer1.view(batch_size, CHANNEL_CNT, -1)).view(batch_size, -1)
        c_layer2 = torch.tanh(self.conv2(inputs.view(batch_size, 1, -1, WV_DIM)))
        p_layer2 = self.pooling2(c_layer2.view(batch_size, CHANNEL_CNT, -1)).view(batch_size, -1)
        c_layer3 = torch.tanh(self.conv3(inputs.view(batch_size, 1, -1, WV_DIM)))
        p_layer3 = self.pooling3(c_layer3.view(batch_size, CHANNEL_CNT, -1)).view(batch_size, -1)
        pooling = torch.cat((p_layer1, p_layer2, p_layer3), dim=1)
        out = torch.tanh(self.linear(pooling))
        return fun.softmax(out, dim=1)


CNN().run()
