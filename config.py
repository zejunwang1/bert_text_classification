# coding: UTF-8

import os
import torch

class Config(object):
    def __init__(self, data_dir):
        assert os.path.exists(data_dir)
        self.train_file = os.path.join(data_dir, "train.txt")
        self.dev_file = os.path.join(data_dir, "dev.txt")
        self.label_file = os.path.join(data_dir, "label.txt")
        assert os.path.isfile(self.train_file)
        assert os.path.isfile(self.dev_file)
        assert os.path.isfile(self.label_file)

        self.saved_model_dir = os.path.join(data_dir, "model")
        self.saved_model = os.path.join(self.saved_model_dir, "bert_model.pth")
        if not os.path.exists(self.saved_model_dir):
            os.mkdir(self.saved_model_dir)

        self.label_list = [label.strip() for label in open(self.label_file, "r", encoding="UTF-8").readlines()]
        self.num_labels = len(self.label_list)

        self.num_epochs = 3
        self.log_batch = 100
        self.batch_size = 128
        self.max_seq_len = 32
        self.require_improvement = 1000

        self.warmup_steps = 0
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.learning_rate = 5e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

