from logging import info
from math import log

from torch import tensor
from torch.nn import functional as fun
from tqdm import tqdm

from interface import Model
from util import TARGET_DIM


class Bayes(Model):
    def __init__(self):
        super().__init__(1, 1, 1, -1, 'index', 1, 'bayes')
        self.prior = [0] * TARGET_DIM
        self.posterior = []

    def do_training(self, train_data, _):
        info('Training...')
        bags = []
        vocab_size = len(self.vocab)
        for _ in range(TARGET_DIM):
            bags.append([0] * vocab_size)
        for _, target, sample in tqdm(train_data):
            bag_id = target.argmax().item()
            for word_idx in sample.view(-1):
                bags[bag_id][word_idx.item()] += 1
            self.prior[bag_id] += 1
        self.prior = [log(cnt) for cnt in self.prior]
        for bag in bags:
            total_len = sum(bag)
            self.posterior.append([log((cnt + 1) / (total_len + vocab_size)) for cnt in bag])

    def forward(self, inputs, _):
        inputs = inputs.numpy().tolist()
        result = [x for x in self.prior]
        for i in range(TARGET_DIM):
            for word_idx in inputs:
                if word_idx != 0:
                    result[i] += self.posterior[i][word_idx]
        return fun.softmax(tensor(result), dim=0)


Bayes().run()
