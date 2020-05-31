import abc
import os
import random
from logging import info
from math import floor

import torch
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from torch import nn, optim, tensor
from torch.nn import functional as fun
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from util import stopwords, DatasetWrapper, word_to_vec


class Model(nn.Module):

    def __init__(self, total_epoch: int, lr: float, gamma: float, wps: int, policy: str, batch_size: int, name: str):
        super().__init__()
        self.total_epoch = total_epoch
        self.lr = lr
        self.gamma = gamma
        self.wps = wps
        self.policy = policy
        self.batch_size = batch_size
        self.name = name
        self.vocab = {'': 0}
        assert policy in ['full', 'average', 'index']

    def encode(self, line: str, training: bool) -> (str, torch.Tensor, torch.Tensor):
        ts, label, text = line.strip().split('\t')

        label = [int(x.split(':')[1]) for x in label.split()[1:]]
        label = fun.softmax(tensor(label, dtype=torch.float32), dim=0)

        text = [word for word in text.split() if word not in stopwords]
        if self.wps > 0:
            text = text[:self.wps]
            text.extend([''] * (self.wps - len(text)))

        if self.policy == 'index':
            sentence = []
            for word in text:
                if word not in self.vocab:
                    if training:
                        idx = len(self.vocab)
                        self.vocab[word] = idx
                        sentence.append(idx)
                    else:
                        sentence.append(0)
                else:
                    sentence.append(self.vocab[word])
            text = tensor(sentence, dtype=torch.long)
        else:
            text = torch.cat([word_to_vec[word].view(1, -1) for word in text])
            if self.policy == 'average':
                text = text.mean(dim=0)
        return ts, label, text

    @abc.abstractmethod
    def forward(self, inputs, batch_size: int):
        pass

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data)

    def load_train(self):
        info('Loading train data...')
        with open('../data/train_data.txt', encoding='utf-8') as f:
            lines = f.readlines()
            line_cnt = len(lines)
            lines = lines[:(line_cnt // self.batch_size) * self.batch_size]
            train_data = [self.encode(line, True) for line in tqdm(lines)]
            train_data = DataLoader(DatasetWrapper(train_data), self.batch_size, True)
        return train_data

    def load_test(self):
        info('Loading test data...')
        with open('../data/test_data.txt', encoding='utf-8') as f:
            test_data = [self.encode(line, False) for line in tqdm(f.readlines())]
        return test_data

    def do_training(self, train_data, validate_data):
        info('Training...')
        with open('../results/%s_loss.txt' % self.name, 'w', encoding='utf-8') as f:
            loss_function = nn.MSELoss()
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
            self.apply(self.init_weights)
            for _ in tqdm(range(self.total_epoch)):
                train_loss = 0
                for _, target, sample in train_data:
                    self.zero_grad()
                    loss = loss_function(self(sample, self.batch_size), target)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                with torch.no_grad():
                    validate_hit = 0
                    for _, target, sample in validate_data:
                        if self(sample, 1).argmax().item() == target.argmax().item():
                            validate_hit += 1
                    f.write("%s, %s\n" % (train_loss, validate_hit / len(validate_data)))
                scheduler.step(None)

    def do_testing(self, test_data):
        info('Testing...')
        with open('../results/%s_result.txt' % self.name, 'w', encoding='utf-8') as f:
            with torch.no_grad():
                predict_list = []
                target_list = []
                correlation_list = []
                for ts, target, sample in tqdm(test_data):
                    predict = self(sample, 1)
                    predict_max = predict.argmax().item()
                    target_max = target.argmax().item()
                    correlation = pearsonr(target.view(-1), predict.view(-1))[0]
                    predict_list.append(predict_max)
                    target_list.append(target_max)
                    correlation_list.append(correlation)
                    f.write('%s: %s %f\n' % (ts, predict.view(-1), correlation))
        return target_list, predict_list, correlation_list

    @staticmethod
    def do_scoring(test_data, score: ([int], [int], [float])):
        target_list, predict_list, correlation_list = score
        hit = [target_list[i] == predict_list[i] for i in range(len(test_data))].count(True)
        info('Accuracy: %d/%d=%f' % (hit, len(test_data), hit / len(test_data)))
        info('F1_macro: %f' % f1_score(target_list, predict_list, average='macro'))
        info('F1_micro: %f' % f1_score(target_list, predict_list, average='micro'))
        info('Weighted: %f' % f1_score(target_list, predict_list, average='weighted'))
        average_correlation = sum(correlation_list) / len(correlation_list)
        info('The following is the percentage distribution of the correlation coefficient.')
        correlation_list = [floor(c * 5) for c in correlation_list]
        info('[0.8,1.0]: %d' %
             round((correlation_list.count(4) + correlation_list.count(5)) * 100 / len(correlation_list)))
        for i in range(3, -6, -1):
            info('[%s,%s): %d' % (i / 5, (i + 1) / 5, round(correlation_list.count(i) * 100 / len(correlation_list))))
        info('Average: %f' % average_correlation)

    def run(self):
        model_file = '../models/%s.pt' % self.name
        pre_trained = os.path.exists(model_file)
        if pre_trained:
            info('Pre-trained model found. Using %s.' % model_file)
            self.load_state_dict(torch.load(model_file))
            test_data = self.load_test()
        else:
            train_data = self.load_train()
            test_data = self.load_test()
            self.do_training(train_data, random.sample(test_data, len(test_data) // 1))
            torch.save(self.state_dict(), model_file)
        score = self.do_testing(test_data)
        self.do_scoring(test_data, score)


if __name__ == '__main__':
    print('Error: Running this script has no effect.')
