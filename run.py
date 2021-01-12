import datetime
import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd


import torch
import torch.nn.functional as F
import numpy as np
import logging
import csv
import json
import torch.nn as nn

from logging import handlers
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataLoad import BaseDataset
from model_LSTM import Model
from data.data import get_data

def init_logger(filename, when='D', backCount=3,
                fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    logger = logging.getLogger(filename)
    format_str = logging.Formatter(fmt)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)

    return logger

# 定义常量

TRAIN_DATASET = 'data/train.pt'
DEV_DATASET = 'data/eval.pt'


#股票数据
INDEX_CODE='399106'
START_TIME='2010-01-01'
END_TIME='2020-12-25'
DATA_PATH='./data/data'+INDEX_CODE+'.csv'

seq_len = 60
LOG_FILE = 'final{0}_{1}.log'.format(INDEX_CODE,seq_len)
RESULT_DATA='result{0}_{1}.csv'.format(INDEX_CODE,seq_len)


BATCH_SIZE = 10
NUM_EPOCH = 8

HIDDEN_DIM = 256
NUM_CLASS = 1
DROP_OUT = 0.3
LEARNING_RATE= 5e-3

TEST_LEN=20

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = init_logger(filename=LOG_FILE)

lossgen=nn.MSELoss()


def convert_x_to_matrix(x,seq_len):
    x_pre=[]
    for i in range(len(x) - seq_len + 1):
        x_pre.append(x[i:i + seq_len])
    return x_pre

# 获取数据
# GETdata=get_data(INDEX_CODE,START_TIME,DATA_PATH,END_TIME)


# 提取列，处理数据
data= pd.read_csv(DATA_PATH)
data=data[["close","open","high","low","volume"]]
close_ori=data.copy()['close'].tolist()
close_min=data['close'].min()
close_max=data['close'].max()
data=data.apply(lambda x:(x-min(x))/(max(x)-min(x)))
y_label=data['close'][seq_len:].values.tolist()
x_pre=data[:-1].values.tolist()
x_pre=convert_x_to_matrix(x_pre,seq_len)
train_x,train_y=x_pre[:len(x_pre)-TEST_LEN],y_label[:len(x_pre)-TEST_LEN]
test_x,test_y=x_pre[len(x_pre)-TEST_LEN:],y_label[len(x_pre)-TEST_LEN:]





dataset=BaseDataset(train_x,train_y)
test_dataset=BaseDataset(test_x,test_y)
torch.save(dataset,TRAIN_DATASET)
torch.save(test_dataset, DEV_DATASET)

dataset = torch.load(TRAIN_DATASET)

train_size = int(0.85 * len(dataset))
train_data, dev_data = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
dev_dataloader = DataLoader(dev_data, batch_size = BATCH_SIZE, shuffle=True)

model = Model( 5, HIDDEN_DIM, 1 , DROP_OUT,NUM_CLASS)
model.to(DEVICE)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# evaluating
def evaluate(dataloader):
    losses, accs = [], []
    for batch in dataloader:
        idx, sent, label = map(lambda x: x.to(DEVICE), batch)

        logits = model(sent)
        label = label.reshape(len(idx), NUM_CLASS)
        loss = lossgen(logits, label)
        logits = (logits * (close_max - close_min) + close_min)
        label = (label * (close_max - close_min) + close_min)

        acc=(abs((logits - label) / label)).sum().float() / label.shape[0]
        # acc = (torch.argmax(logits, dim=-1) == label).sum().float() / label.shape[0]

        losses.append(loss.item())
        accs.append(acc.item())
    return losses,accs

# trainning

for epoch in range(NUM_EPOCH):
    logger.info('=' * 100)
    losses, accs = [], []
    pbar = tqdm(total=len(train_dataloader))
    model.train()
    for batch in train_dataloader:
        idx, sent, label = map(lambda x: x.to(DEVICE), batch)

        logits = model(sent)
        label=label.reshape(len(idx),NUM_CLASS)

        loss = lossgen(logits, label)
        logits = (logits * (close_max - close_min) + close_min)
        label = (label * (close_max - close_min) + close_min)
        acc = (abs((logits - label) / label)).sum().float() / label.shape[0]

        # acc = (torch.argmax(logits, dim=-1) == label).sum().float() / label.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append(acc.item())

        pbar.set_description('Epoch: %2d | accuracy：%.6f | Loss: %.3f ' % (epoch,acc.item(), loss.item()))
        pbar.update(1)

    pbar.close()

    # logger.info log
    model.eval()
    dev_loss,dev_acc = evaluate(dev_dataloader)
    logger.info('Training:\t  accuracy：%.6f | Loss: %.6f' % (np.mean(accs),np.mean(losses)))
    logger.info('Evaluating:\t accuracy：%.6f | Loss: %.6f' % (np.mean(dev_acc),np.mean(dev_loss)))
    logger.info('')

# writing result
result = open(RESULT_DATA, 'w', newline='',encoding='utf-8')
result_writer = csv.writer(result)
result_writer.writerow(['idx', 'pred','true'])


test_dataset = torch.load(DEV_DATASET)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

losses = []
accs=[]
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        idx, sent, label = map(lambda x: x.to(DEVICE), batch)
        pred = model(sent)
        label = label.reshape(len(idx), NUM_CLASS)
        pred_ori=(pred*(close_max-close_min)+close_min)
        label_ori=(label*(close_max-close_min)+close_min)
        loss = lossgen(pred_ori, label_ori)
        acc = (abs((pred_ori - label_ori) / label_ori)).sum().float() / label_ori.shape[0]

        pred_ori=sum(pred_ori.cpu().numpy().tolist(),[])
        label_ori=sum(label_ori.cpu().numpy().tolist(),[])

        losses.append(loss.item())
        accs.append(acc.item())

        for i, pred,true in zip(idx, pred_ori,label_ori):
            result_writer.writerow([int(i), pred,true])

logger.info('seq_len= %2d|  batch_size= %2d | test_size=%2d |lr=%.5f after info: Loss(RMSE): %.3f  Accuracy: %.6f' % (seq_len,BATCH_SIZE,TEST_LEN,LEARNING_RATE,sqrt(np.mean(losses)),np.mean(accs)))
# plt.plot(range(len(losses)),losses)
# plt.show()