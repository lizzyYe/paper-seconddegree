import torch
# import nltk
import json
import re
import csv

from collections import Counter
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self,x,y=None):
        self.idxs, self.sents, self.labels = self.convert_data(x, y)

    def convert_data(self, data,y=None):
        idxs, sents, labels = [], [], []

        for i in range(len(data)):
            idxs.append(i)
            sents.append(data[i])
            if y==None:
                labels.append(-1)
            else:
                labels.append(y[i])

        idxs, sents, labels = torch.LongTensor(idxs), torch.Tensor(sents), torch.Tensor(labels)

        return idxs, sents, labels

    def __getitem__(self, index):
        return (self.idxs[index], self.sents[index], self.labels[index])

    def __len__(self):
        return self.sents.shape[0]


