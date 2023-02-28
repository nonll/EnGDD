import argparse
import os, sys, time
sys.path.append("..")
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, \
    precision_score, auc
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from src.cv_123 import CV, index_fea
from src.cal_time import cal_time
from src.cal_metrics import cal_metrics,save_result


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NR')
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--predictor', type=str, default='forest')
    parser.add_argument('--method', type=str, default='DNN')
    parser.add_argument('--cuda', type=str, default=True)
    args = parser.parse_args()
    return args

class DNN(nn.Module):
    def __init__(self, inputNode, hidden, outputNode):
        super().__init__()
        self.liner_1 = nn.Linear(inputNode, hidden)
        self.liner_2 = nn.Linear(hidden, hidden)
        self.drop_layer = nn.Dropout(p=0.25)
        self.liner_2_1 = nn.Linear(hidden, hidden // 2)
        self.liner_3 = nn.Linear(hidden // 2, hidden // 4)
        self.liner_4 = nn.Linear(hidden // 4, outputNode)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        前向传播
        """
        x = self.liner_1(input)
        x = self.relu(x)
        # x = self.drop_layer(x)
        x = self.liner_2(x)
        x = self.relu(x)
        # x = self.drop_layer(x)
        x = self.liner_2_1(x)
        x = self.relu(x)
        # x = self.drop_layer(x)
        x = self.liner_3(x)
        x = self.liner_4(x)
        x = self.sigmoid(x)
        return x

    @classmethod
    def get_model(cls, input, hidden, output):
        model = DNN(input, hidden, output)
        return model


class Trainer_DNN(object):
    def __init__(self, model,lr,device):
        self.model = model
        self.device=device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,weight_decay=0.001)
        self.loss_fn = torch.nn.BCELoss().to(device=device)

    def train(self, train_loader):
        loss_total = 0
        # print('train-->',self.model.named_parameters())
        N = len(train_loader)
        size = len(train_loader.dataset)

        for batch, (x, y) in enumerate(train_loader):
            x, y = x.to(device=self.device), y.to(device=self.device)
            self.model.train(mode=True)
            x = torch.as_tensor(x, dtype=torch.float32)
            out = self.model(x)
            out = out.view(-1)
            loss = self.loss_fn(out, y.float()).to(device=self.device)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
            # print(loss.item())
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        return loss_total


class Tester_DNN(object):
    def __init__(self,model,device):
        self.model = model
        self.device=device
        self.loss_fn = torch.nn.BCELoss().to(device=device)

    def test(self, test_loader):
        N = len(test_loader)
        size = len(test_loader.dataset)
        num_batches = len(test_loader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            P, C = [], []
            for X, y in test_loader:
                X, y = X.to(device=self.device), y.to(device=self.device)
                X = torch.as_tensor(X, dtype=torch.float32)
                self.model.eval()
                pred = self.model(X)
                pred = pred.view(-1)
                test_loss += self.loss_fn(pred, y.float()).item()
                pred ,y = pred.cpu() ,y.cpu()
                correct += (pred.round() == y).type(torch.float).sum().item()
                C.append(y)
                P.append(pred)
        test_loss /= num_batches
        correct /= size
        print(f"\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        y, pred = np.concatenate(C), np.concatenate(P)     
        return y, pred
        
    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')


