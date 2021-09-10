# coding: UTF-8

import time
import torch
import random
from tqdm import tqdm
from datetime import timedelta

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

class DataProcessor(object):
    def __init__(self, path, device, tokenizer, batch_size, max_seq_len, seed):
        self.seed = seed
        self.device = device
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.data = self.load(path)

        self.index = 0
        self.residue = False
        self.num_samples = len(self.data[0])
        self.num_batches = self.num_samples // self.batch_size
        if self.num_samples % self.batch_size != 0:
            self.residue = True

    def load(self, path):
        contents = []
        labels = []
        with open(path, mode="r", encoding="UTF-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:    continue
                if line.find('\t') == -1:   continue
                content, label = line.split("\t")
                contents.append(content)
                labels.append(int(label))
        #random shuffle
        index = list(range(len(labels)))
        random.seed(seed)
        random.shuffle(index)
        contents = [contents[_] for _ in index]
        labels = [labels[_] for _ in index]
        return (contents, labels)

    def __next__(self):
        if self.residue and self.index == self.num_batches:
            batch_x = self.data[0][self.index * self.batch_size: self.num_samples]
            batch_y = self.data[1][self.index * self.batch_size: self.num_samples]
            batch = self._to_tensor(batch_x, batch_y)
            self.index += 1
            return batch
        elif self.index >= self.num_batches:
            self.index = 0
            raise StopIteration
        else:
            batch_x = self.data[0][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batch_y = self.data[1][self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batch = self._to_tensor(batch_x, batch_y)
            self.index += 1
            return batch

    def _to_tensor(self, batch_x, batch_y):
        inputs = self.tokenizer.batch_encode_plus(
            batch_x,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation="longest_first",
            return_tensors="pt")
        inputs = inputs.to(self.device)
        labels = torch.LongTensor(batch_y).to(self.device)
        return (inputs, labels)

    def __iter__(self):
        return self
    
    def __len__(self):
        if self.residue:
            return self.num_batches + 1
        else:
            return self.num_batches
