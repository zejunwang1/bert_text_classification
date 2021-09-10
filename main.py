# coding: UTF-8

import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from train import train
from config import Config
from preprocess import DataProcessor, get_time_dif
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

parser = argparse.ArgumentParser(description="Bert Chinese Text Classification")
parser.add_argument("--mode", type=str, required=True, help="train/demo/predict")
parser.add_argument("--data_dir", type=str, default="./data", help="training data and saved model path")
parser.add_argument("--pretrained_bert_dir", type=str, default="./pretrained_bert", help="pretrained bert model path")
parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
parser.add_argument("--input_file", type=str, default="./data/input.txt", help="input file to be predicted")
args = parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    set_seed(args.seed)
    config = Config(args.data_dir)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_dir)
    bert_config = BertConfig.from_pretrained(args.pretrained_bert_dir, num_labels=config.num_labels)
    model = BertForSequenceClassification.from_pretrained(
        os.path.join(args.pretrained_bert_dir, "pytorch_model.bin"),
        config=bert_config
    )
    model.to(config.device)

    if args.mode == "train":
        print("loading data...")
        start_time = time.time()
        train_iterator = DataProcessor(config.train_file, config.device, tokenizer, config.batch_size, config.max_seq_len, args.seed)
        dev_iterator = DataProcessor(config.dev_file, config.device, tokenizer, config.batch_size, config.max_seq_len, args.seed)
        time_dif = get_time_dif(start_time)
        print("time usage:", time_dif)

        # train
        train(model, config, train_iterator, dev_iterator)
    
    elif args.mode == "demo":
        model.load_state_dict(torch.load(config.saved_model))
        model.eval()
        while True:
            sentence = input("请输入文本:\n")
            inputs = tokenizer(
                sentence, 
                max_length=config.max_seq_len,
                truncation="longest_first",
                return_tensors="pt")
            inputs = inputs.to(config.device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs[0]
                label = torch.max(logits.data, 1)[1].tolist()
                print("分类结果:" + config.label_list[label[0]])
            flag = str(input("continue? (y/n):"))
            if flag == "Y" or flag == "y":
                continue
            else:
                break
    else:
        model.load_state_dict(torch.load(config.saved_model))
        model.eval()

        text = []
        with open(args.input_file, mode="r", encoding="UTF-8") as f:
            for line in tqdm(f):
                sentence = line.strip()
                if not sentence:    continue
                text.append(sentence)

        num_samples = len(text)
        num_batches = (num_samples - 1) // config.batch_size + 1
        for i in range(num_batches):
            start = i * config.batch_size
            end = min(num_samples, (i + 1) * config.batch_size)
            inputs = tokenizer.batch_encode_plus(
                text[start: end],
                padding=True,
                max_length=config.max_seq_len,
                truncation="longest_first",
                return_tensors="pt")
            inputs = inputs.to(config.device)

            outputs = model(**inputs)
            logits = outputs[0]

            preds = torch.max(logits.data, 1)[1].tolist()
            labels = [config.label_list[_] for _ in preds]
            for j in range(start, end):
                print("%s\t%s" % (text[j], labels[j - start]))
                

if __name__ == "__main__":
    main()
