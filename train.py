# coding: UTF-8

from typing import Iterator
from transformers import AdamW, get_linear_schedule_with_warmup
from preprocess import get_time_dif
from sklearn import metrics
import time
import torch
import numpy as np

def eval(model, config, iterator, flag=False):
    model.eval()

    total_loss = 0
    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    with torch.no_grad():
        for batch, labels in iterator:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                labels=labels)

            loss = outputs[0]
            logits = outputs[1]

            total_loss += loss
            true = labels.data.cpu().numpy()
            pred = torch.max(logits.data, 1)[1].cpu().numpy()
            all_labels = np.append(all_labels, true)
            all_preds = np.append(all_preds, pred)
    
    acc = metrics.accuracy_score(all_labels, all_preds)
    if flag:
        report = metrics.classification_report(all_labels, all_preds, target_names=config.label_list, digits=4)
        confusion = metrics.confusion_matrix(all_labels, all_preds)
        return acc, total_loss / len(iterator), report, confusion
    return acc, total_loss / len(iterator)


def test(model, config, iterator):
    model.load_state_dict(torch.load(config.saved_model))
    start_time = time.time()
    acc, loss, report, confusion = eval(model, config, iterator, flag=True)
    msg = "Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}"
    print(msg.format(loss, acc))
    print("Precision, Recall and F1-Score...")
    print(report)
    print("Confusion Matrix...")
    print(confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def train(model, config, train_iterator, dev_iterator):
    model.train()
    start_time = time.time()

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    param_optimizer = model.named_parameters()
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = len(train_iterator) * config.num_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=t_total)

    total_batch = 0
    last_improve = 0
    break_flag = False
    best_dev_loss = float('inf')
    for epoch in range(config.num_epochs):
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        for _, (batch, labels) in enumerate(train_iterator):

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                labels=labels)
            
            loss = outputs[0]
            logits = outputs[1]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            scheduler.step() 

            if total_batch % config.log_batch == 0:
                true = labels.data.cpu()
                pred = torch.max(logits.data, 1)[1].cpu()
                acc = metrics.accuracy_score(true, pred)
                dev_acc, dev_loss = eval(model, config, dev_iterator)
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    torch.save(model.state_dict(), config.saved_model)
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ""

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Batch Train Loss: {1:>5.2}, Batch Train Acc: {2:>6.2%}, Val Loss: {3:>5.2}, Val Acc: {4:>6.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), acc, dev_loss, dev_acc, time_dif, improve))
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No improvement for a long time, auto-stopping...")
                break_flag = True
                break
        if break_flag:
            break
    
    test(model, config, dev_iterator)

