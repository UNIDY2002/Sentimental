import logging
import os

import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vectors

WV_DIM = 300
TARGET_DIM = 8

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

stopwords = {line.strip() for line in open('../data/stopwords.txt', encoding='utf-8')}

if not os.path.exists('../data/sgns.merge.word') and not os.path.exists('../data/sgns.merge.word.pt'):
    logging.error('Neither sgns.merge.word nor sgns.merge.word.pt was found. '
                  'Please refer to README.md for more information.')
    raise FileNotFoundError

word_to_vec = Vectors('../data/sgns.merge.word', cache='../data')


class DatasetWrapper(Dataset):
    def __init__(self, src: [(str, torch.Tensor, torch.Tensor)]):
        self.src = src

    def __getitem__(self, index):
        return self.src[index]

    def __len__(self):
        return len(self.src)


if __name__ == '__main__':
    print(word_to_vec['中文'])
    print(word_to_vec['词'])
    print(word_to_vec['向量'])
