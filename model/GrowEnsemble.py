#!/usr/bin/env python
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import argparse
import copy
import time,sys,os
sys.path.append("..") 
import torch
import pandas as pd
import random
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
from GrowNet.mlp import MLP_2HL, MLP_3HL, MLP_4HL, DNN
from GrowNet.dynamic_net import DynamicNet
from torch.optim import Adam
from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, \
    precision_score
from src.cv_123 import CV, index_fea
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NR')
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--feat_d', type=int, default=200)
    parser.add_argument('--hidden_d', type=int, default=32)
    parser.add_argument('--boost_rate', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num_nets', type=int, default=20)
    parser.add_argument('--method', type=str, default='gb_ense')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs_per_stage', type=int, default=1)
    parser.add_argument('--correct_epoch', type=int, default=1)
    parser.add_argument('--L2', type=float, default='0.001')
    parser.add_argument('--sparse', default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--normalization', default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--model_order', default='second', type=str)

    parser.add_argument('--cuda', type=str, default=True)
    args = parser.parse_args()
    return args


def get_optim(params, lr, weight_decay):
    optimizer = Adam(params, lr, weight_decay=weight_decay)
    return optimizer


def accuracy(net_ensemble, test_loader):
    correct = 0
    total = 0
    loss = 0
    for x, y in test_loader:
        if opt.cuda:
            x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            middle_feat, out = net_ensemble.forward(x)
        correct += (torch.sum(y[out > 0.] > 0) + torch.sum(y[out < .0] < 0)).item()
        total += y.numel()
    return correct / total

def logloss(net_ensemble, test_loader):
    loss = 0
    total = 0
    loss_f = nn.BCEWithLogitsLoss()  
        if opt.cuda:
            x, y = x.cuda(), y.cuda().view(-1, 1)
        x = torch.tensor(x, dtype=torch.float32)
        y = (y + 1) / 2
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
        loss += loss_f(out, y)
        total += 1
    return loss / total

def cal_m(net_ensemble, test_loader,args):
    actual = []
    posterior = []
    for x, y in test_loader:
        x = torch.tensor(x, dtype=torch.float32)
        if opt.cuda:
            x = x.cuda()
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        prob = 1.0 - 1.0 / torch.exp(out)  
        posterior.extend(prob)
        actual.extend(y.numpy().tolist())
    posterior = np.array(posterior)
    print(posterior)
    posterior[posterior > 1] = 1
    posterior[posterior < 0] = 0
    pr = copy.deepcopy(posterior)
    pr[pr >= 0.5] = 1
    pr[pr < 0.5] = 0
    fpr, tpr, threshold = roc_curve(actual, posterior)
    pre, rec_, _ = precision_recall_curve(actual, posterior)
    acc = accuracy_score(actual, pr)
    rec = recall_score(actual, pr)
    f1 = f1_score(actual, pr)
    Pre = precision_score(actual, pr)
    au = auc(fpr, tpr)
    apr = auc(rec_, pre)

    print('Precision is :{}'.format(Pre))
    print('Recall is :{}'.format(rec))
    print("ACC is: {}".format(acc))
    print("F1 is: {}".format(f1))
    print("AUC is: {}".format(au))
    print('AUPR is :{}'.format(apr))
    print("-----------------------------------")
    ss=(f'Precision: {Pre:.4f}\nRecall: {rec:.4f}\nACC: {acc:.4f}\nF1: {f1:.4f}\nAUC: {au:.4f}\nAUPR: {apr:.4f}\n\n')
    return ss

def auc_score(net_ensemble, test_loader):
    actual = []
    posterior = []
    for x, y in test_loader:
        x = torch.tensor(x, dtype=torch.float32)
        if opt.cuda:
            x = x.cuda()
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        prob = 1.0 - 1.0 / torch.exp(out)  
        prob = prob.cpu().numpy().tolist()
        posterior.extend(prob)
        actual.extend(y.numpy().tolist())
    posterior = np.array(posterior)
    posterior[posterior > 1] = 1
    posterior[posterior < 0] = 0
    fpr, tpr, threshold = roc_curve(actual, posterior)
    au = auc(fpr, tpr)
    return au


def init_gbnn(train):
    positive = negative = 0
    for i in range(len(train)):
        if train[i][1] > 0:
            positive += 1
        else:
            negative += 1
    blind_acc = max(positive, negative) / (positive + negative)
    print(f'Blind accuracy: {blind_acc}')
    return blind_acc

def kfold(data, k, row=0, col=0, cv=3):
    dlen = len(data)
    if cv == 1:
        lens = row
    elif cv == 2:
        lens = col
    else:
        lens = dlen
    d = list(range(lens))
    random.shuffle(d)
    test_n = lens // k
    n = lens % k
    test_res = []
    train_res = []
    for i in range(n):
        test = d[i * (test_n + 1):(i + 1) * (test_n + 1)]
        train = list(set(d) - set(test))
        test_res.append(test)
        train_res.append(train)
    for i in range(n, k):
        test = d[i * test_n + n:(i + 1) * test_n + n]
        train = list(set(d) - set(test))
        test_res.append(test)
        train_res.append(train)
    if cv == 3:
        return train_res, test_res
    else:
        train_s = []
        test_s = []
        for i in range(k):
            train_ = []
            test_ = []
            for j in range(dlen):
                if data[j][cv - 1] in test_res[i]:
                    test_.append(j)
                else:
                    train_.append(j)
            train_s.append(train_)
            test_s.append(test_)
        return train_s, test_s

def save_result(name,opt):
    file = f'./output/cv{opt.cv}/n_res/{name}_res.txt'
    pre, rec,acc,f1,auc,aupr=[],[],[],[],[],[]
    with open(file, 'r') as f:
        for line in f.readlines():
            if "Precision"  in line:
                pre.append(float(line.split(': ')[1][:-1]))    
            if 'Recall' in line:
                rec.append(float(line.split(': ')[1][:-1]))
            if 'ACC' in line:
                acc.append(float(line.split(': ')[1][:-1]))
            if 'F1' in line:
                f1.append(float(line.split(': ')[1][:-1]))
            if 'AUC' in line:
                auc.append(float(line.split(': ')[1][:-1]))
            if 'AUPR' in line:
                aupr.append(float(line.split(': ')[1][:-1]))

    pre, rec,acc,f1,auc,aupr = np.array(pre),np.array(rec),np.array(acc),np.array(f1),np.array(auc),np.array(aupr)
    

    with open(f'./output/cv{opt.cv}/average/{name}.txt',mode='a')as f:
        f.write(f'{name}\n{"="*50}\ntime:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n{opt}\n{"="*50}\n')
        
        f.write(f'Precision: {pre.mean():.4f}±{np.std(pre):.4f}\n')
        f.write(f'Recall: {rec.mean():.4f}±{np.std(rec):.4f}\n')
        f.write(f'ACC: {acc.mean():.4f}±{np.std(acc):.4f}\n')
        f.write(f'F1: {f1.mean():.4f}±{np.std(f1):.4f}\n')
        f.write(f'AUC: {auc.mean():.4f}±{np.std(auc):.4f}\n')
        f.write(f'AUPR: {aupr.mean():.4f}±{np.std(aupr):.4f}\n\n')
        f.close()


def cal_time(seconds):
 
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h>0:
        print(f'The program is running time: {h:.0f}h{m:.0f}m{s:.0f}s')
    elif m>0:
        print(f'The program is running time: {m:.0f}m{s:.0f}s')
    else:
        print(f'The program is running time: {s:.4f}s')
 

if __name__ == "__main__":
    st = time.time()
    opt = parse_arguments()
    name = opt.dataset
    result=[]
    path = f"/data/wangyz/PycharmProjects/AE_gc/Feature/Preprocess/{name}/{name}_matrix.csv"
    df = pd.read_csv(path, header=None)
    print(df.shape)
    path_fea = f"/data/wangyz/PycharmProjects/AE_gc/Feature/{name}.csv"
    fea = pd.read_csv(path_fea, header=None)
    print(fea.shape)
    trains, tests = CV.get_cv(cv=opt.cv, n_repeats=5, inc_matrix=df)
    for i in range(len(trains)):
        lr, L2 = opt.lr, opt.L2

        train_index = index_fea(trains[i], col=df.shape[1])
        test_index = index_fea(tests[i], col=df.shape[1])
        train, test = fea.loc[train_index.reshape(-1)], fea.loc[test_index.reshape(-1)]
        # print(train.shape, test.shape)
        train, test = train.to_numpy(), test.to_numpy()
        val = copy.deepcopy(train)
        train = TensorDataset(torch.from_numpy(
            train[:, :-1]), torch.from_numpy((train[:, -1]).astype('int')))
        test = TensorDataset(torch.from_numpy(
            test[:, :-1]), torch.from_numpy((test[:, -1]).astype('int')))
        val = TensorDataset(torch.from_numpy(
            val[:, :-1]), torch.from_numpy((val[:, -1]).astype('int')))
        print(opt.dataset + ' training and test datasets are loaded!')
        train_loader = DataLoader(train, opt.batch_size, shuffle=True, drop_last=False, num_workers=2)
        test_loader = DataLoader(test, opt.batch_size, shuffle=False, drop_last=False, num_workers=2)
        # if opt.cv:
        val_loader = DataLoader(val, opt.batch_size, shuffle=True, drop_last=False, num_workers=2)
        # For CV use
        best_score = 0
        val_score = best_score
        best_stage = opt.num_nets - 1
        c0 = init_gbnn(train)
        net_ensemble = DynamicNet(c0, opt.boost_rate)
        loss_f1 = nn.MSELoss(reduction='none')
        loss_f2 = nn.BCEWithLogitsLoss(reduction='none')
        loss_models = torch.zeros((opt.num_nets, 3))

        all_ensm_losses = []
        all_ensm_losses_te = []
        all_mdl_losses = []
        dynamic_br = []

        for stage in range(opt.num_nets):
            t0 = time.time()
            model = MLP_2HL.get_model(stage, opt)  
            if opt.cuda:
                model.cuda()

            optimizer = get_optim(model.parameters(), lr, L2)
            net_ensemble.to_train()  

            stage_mdlloss = []
            for epoch in range(opt.epochs_per_stage):
                for l, (x, y) in enumerate(train_loader):
                    if opt.cuda:
                        x, y = x.cuda(), y.cuda().view(-1, 1)
                    x = torch.tensor(x, dtype=torch.float32)
                    middle_feat, out = net_ensemble.forward(x)
                    out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                    if opt.model_order == 'first':
                        grad_direction = y / (1.0 + torch.exp(y * out))
                    else:
                        h = 1 / ((1 + torch.exp(y * out)) * (1 + torch.exp(-y * out)))
                        grad_direction = y * (1.0 + torch.exp(-y * out))
                        out = torch.as_tensor(out)
                        nwtn_weights = (torch.exp(out) + torch.exp(-out)).abs()
                    _, out = model(x, middle_feat)
                    out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                    loss = loss_f1(net_ensemble.boost_rate * out, grad_direction)  # T
                    loss = loss * h
                    loss = loss.mean()
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    stage_mdlloss.append(loss.item())

            net_ensemble.add(model)
            sml = np.mean(stage_mdlloss)

            stage_loss = []
            lr_scaler = 2
            # fully-corrective step
            if stage != 0:
                # Adjusting corrective step learning rate
                if stage % 5 == 0:  # 15 -> 10
                    lr /= 2
                    L2 /= 2
                optimizer = get_optim(net_ensemble.parameters(), lr / lr_scaler, L2)
                for _ in range(opt.correct_epoch):
                    for ls, (x, y) in enumerate(train_loader):
                        if opt.cuda:
                            x, y = x.cuda(), y.cuda().view(-1, 1)
                        x = torch.tensor(x, dtype=torch.float32)
                        _, out = net_ensemble.forward_grad(x)
                        out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                        y = (y + 1.0) / 2.0
                        loss = loss_f2(out, y).mean()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        stage_loss.append(loss.item())

            sl_te = logloss(net_ensemble, test_loader)
            # Store dynamic boost rate
            dynamic_br.append(net_ensemble.boost_rate.item())
            elapsed_tr = time.time() - t0
            sl = 0
            if stage_loss != []:
                sl = np.mean(stage_loss)

            all_ensm_losses.append(sl)
            all_ensm_losses_te.append(sl_te)
            all_mdl_losses.append(sml)
            print(
                f'Stage - {stage}, training time: {elapsed_tr: .1f} sec, Training Loss: {sl: .4f}, Test Loss: {sl_te: .4f}')   
            if opt.cuda:
                net_ensemble.to_cuda()
            net_ensemble.to_eval()  
            # Train
            print('Acc results from stage := ' + str(stage) + '\n')
            # AUC
            # if opt.cv:
            val_score = auc_score(net_ensemble, val_loader)
            if val_score > best_score:
                best_score = val_score
                best_stage = stage

            test_score = auc_score(net_ensemble, test_loader)
            print(f'Stage: {stage}, AUC@Val: {val_score:.4f}, AUC@Test: {test_score:.4f}')

            loss_models[stage, 1], loss_models[stage, 2] = val_score, test_score

        val_auc, te_auc = loss_models[best_stage, 1], loss_models[best_stage, 2]
        print(f'Best validation stage: {best_stage},  AUC@Val: {val_auc:.4f}, final AUC@Test: {te_auc:.4f}')
        result1=cal_m(net_ensemble, test_loader,opt)
        result.append(result1)
    with open(f'./output/cv{opt.cv}/n_res/{name}_res.txt',mode='w') as f:
        for i in result:
            f.write(i)
            
    save_result(name,opt)
    cal_time(time.time()-st)
                      