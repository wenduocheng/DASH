import os
import argparse
import random
import math # 
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from timeit import default_timer
from task_configs import get_data
from task_utils import accuracy

torch.cuda.set_device(1)
print(torch.cuda.is_available())
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", DEVICE)

dataset= "human_ensembl_regulatory" # "human_nontata_promoters", "human_enhancers_cohn"

class DeepSEA(torch.nn.Module):
    def __init__(self, ):
        super(DeepSEA, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=5, out_channels=320, kernel_size=8, padding = 4)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8, padding = 4)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8, padding = 4)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.Linear = nn.Linear(960 * 51, 3) # 51: "human_ensembl_regulatory"
        # self.Linear = nn.Linear(960 * 17, 2)

    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop2(x)
        # print(x.size())
        x = self.flatten(x)
        x = self.Linear(x)

        return x


model = DeepSEA().to(DEVICE) 
print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, weight_decay=0) 

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=0)

loss = torch.nn.CrossEntropyLoss()

EPOCH=20


# get data
train_loader, val_loader, test_loader, n_train, n_val, n_test, _ = get_data(dataset, batch_size=32, arch="wrn", valid_split=False)
for batch in train_loader: #
    x, y = batch
    print(x.size())
    print(y.size())
    break

def evaluate(model, loader, loss, metric, n_eval):
    model.eval()
    eval_loss, eval_score = 0, 0

    with torch.no_grad():
        for i, data in enumerate(loader):
            x, y = data
                                
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)

            eval_loss += loss(out, y).item()
            eval_score += metric(out, y).item()
      
        eval_loss /= n_eval
        eval_score /= n_eval
    return eval_loss, eval_score


def train_one_epoch(model, optimizer, scheduler, loader, loss, temp):    

    model.train()
                    
    train_loss = 0
    optimizer.zero_grad()

    for i, data in enumerate(loader):
        x, y = data 
        # print(x[0][0])
        
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)

        l = loss(out, y)
        l.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.9)
       
        optimizer.step()
        optimizer.zero_grad()

        train_loss += l.item()

        if i >= temp - 1:
            break


    return train_loss / temp


print("\n------- Start Training --------")
# training and validating


metric = accuracy

train_time, train_score, train_losses = [], [], []

for ep in range(EPOCH):
    # train
    time_start = default_timer()
    train_loss = train_one_epoch(model, optimizer, scheduler, train_loader, loss, n_train)
    train_time_ep = default_timer() -  time_start 
    # val    
    val_loss, val_score = evaluate(model, val_loader, loss, metric, n_val)
    
    train_losses.append(train_loss)
    train_score.append(val_score)
    train_time.append(train_time_ep)

    scheduler.step(val_loss)
    
    print("[train", "full", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (train_time[-1]), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % np.max(train_score))
 


