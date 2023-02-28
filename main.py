#!/usr/bin/env python
import warnings
import deepforest
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
from model import DNN 
from deepforest import CascadeForestClassifier
from model.GrowNet.mlp import MLP_2HL, MLP_3HL, MLP_4HL
from model.GrowNet.dynamic_net import DynamicNet
from src.cal_metrics import cal_metrics,save_result
from torch.optim import Adam
from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, \
    precision_score
from src.cv_123 import CV, index_fea

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NR')
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--feat_d', type=int, default=200)
    parser.add_argument('--hidden_d', type=int, default=32)
    parser.add_argument('--boost_rate', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num_nets', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--method', type=str, default='ense')
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
        if args.cuda:
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
    for x, y in test_loader:
        if args.cuda:
            x, y = x.cuda(), y.cuda().view(-1, 1)
        x = torch.tensor(x, dtype=torch.float32)
        y = (y + 1) / 2
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
        loss += loss_f(out, y)
        total += 1
    return loss / total

def cal_m(net_ensemble, test_loader):
    actual = []
    posterior = []
    for x, y in test_loader:
        x = torch.tensor(x, dtype=torch.float32)
        if args.cuda:
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
        if args.cuda:
            x = x.cuda()
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        prob = 1.0 - 1.0 / torch.exp(out)  # Why not using the scores themselve than converting to prob
        prob = prob.cpu().numpy().tolist()
        posterior.extend(prob)
        actual.extend(y.numpy().tolist())
    posterior = np.array(posterior)
    posterior[posterior > 1] = 1
    posterior[posterior < 0] = 0
    fpr, tpr, threshold = roc_curve(actual, posterior)
    au = auc(fpr, tpr)
    return au

def en_proba_score(net_ensemble, test_loader):
    actual = []
    posterior = []
    for x, y in test_loader:
        x = torch.tensor(x, dtype=torch.float32)
        if args.cuda:
            x = x.cuda()
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        prob = 1.0 - 1.0 / torch.exp(out)  # Why not using the scores themselve than converting to prob
        prob = prob.cpu().numpy().tolist()
        posterior.extend(prob)
        actual.extend(y.numpy().tolist())
    posterior = np.array(posterior)
    posterior[posterior > 1] = 1
    posterior[posterior < 0] = 0
    return posterior

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

def cal_time(seconds):
    # Calculate running time
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
    args= parse_arguments()
    lr = args.lr
    name = args.dataset
    result=[]
    path = f"./Feature/Preprocess/{name}/{name}_matrix.csv"
    df = pd.read_csv(path, header=None)
    path_fea = f"./Feature/{name}.csv"
    fea = pd.read_csv(path_fea, header=None)
    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
    trains, tests = CV.get_cv(cv=args.cv, n_repeats=1, inc_matrix=df)
    for i in range(len(trains)):
        lr, L2 = args.lr, args.L2
        train_index = index_fea(trains[i], col=df.shape[1])
        test_index = index_fea(tests[i], col=df.shape[1])
        train, test = fea.loc[train_index.reshape(-1)], fea.loc[test_index.reshape(-1)]
        train, test = train.to_numpy(), test.to_numpy()
        X_train, y_train = train[:, :-1], train[:, -1]
        X_test, y_test = test[:, :-1], test[:, -1]
        val = copy.deepcopy(train)
        train = TensorDataset(torch.from_numpy(
            train[:, :-1]), torch.from_numpy((train[:, -1]).astype('int')))
        test = TensorDataset(torch.from_numpy(
            test[:, :-1]), torch.from_numpy((test[:, -1]).astype('int')))
        val = TensorDataset(torch.from_numpy(
            val[:, :-1]), torch.from_numpy((val[:, -1]).astype('int')))
        print(args.dataset + ' training and test datasets are loaded!')
        train_loader = DataLoader(train, args.batch_size, shuffle=True, drop_last=False, num_workers=2)
        test_loader = DataLoader(test, args.batch_size, shuffle=False, drop_last=False, num_workers=2)
        # if args.cv:
        val_loader = DataLoader(val, args.batch_size, shuffle=True, drop_last=False, num_workers=2)
        # DF-->train      
        Deepforest = CascadeForestClassifier(n_estimators=5, predictor='forest')
        Deepforest.fit(X_train, y_train)
        # DNN-->train
        # init
        dnn = DNN.DNN.get_model(200, 64, 1)
        dnn = dnn.to(device)
        trainer = DNN.Trainer_DNN(dnn,args.lr,device=device)
        tester = DNN.Tester_DNN(dnn,device=device)
        for t in range(args.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            # train DNN
            loss_train = trainer.train(train_loader)
        # For CV use
        best_score = 0
        val_score = best_score
        best_stage = args.num_nets - 1
        c0 = init_gbnn(train)
        net_ensemble = DynamicNet(c0, args.boost_rate)
        loss_f1 = nn.MSELoss(reduction='none')
        loss_f2 = nn.BCEWithLogitsLoss(reduction='none')
        loss_models = torch.zeros((args.num_nets, 3))
        all_ensm_losses = []
        all_ensm_losses_te = []
        all_mdl_losses = []
        dynamic_br = []
        for stage in range(args.num_nets):
            t0 = time.time()
            # Initialize the model_k: f_k(x)
            model = MLP_2HL.get_model(stage, args)
            if args.cuda:
                model.cuda()
            optimizer = get_optim(model.parameters(), lr, L2)
            # Set the models in ensemble net to train mode
            net_ensemble.to_train()  
            stage_mdlloss = []
            for epoch in range(args.epochs_per_stage):
                for l, (x, y) in enumerate(train_loader):
                    if args.cuda:
                        x, y = x.cuda(), y.cuda().view(-1, 1)
                    x = torch.tensor(x, dtype=torch.float32)
                    middle_feat, out = net_ensemble.forward(x)
                    out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                    if args.model_order == 'first':
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
                    # lr_scaler *= 2  # TODO
                    lr /= 2
                    L2 /= 2
                optimizer = get_optim(net_ensemble.parameters(), lr / lr_scaler, L2)
                for _ in range(args.correct_epoch):
                    for ls, (x, y) in enumerate(train_loader):
                        if args.cuda:
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
            if args.cuda:
                net_ensemble.to_cuda()
            # Set the models in ensemble net to eval mode    
            net_ensemble.to_eval()  
            # Train
            print('Acc results from stage := ' + str(stage) + '\n')
            # AUC
            # if args.cv:
            val_score = auc_score(net_ensemble, val_loader)
            if val_score > best_score:
                best_score = val_score
                best_stage = stage
            test_score = auc_score(net_ensemble, test_loader)
            print(f'Stage: {stage}, AUC@Val: {val_score:.4f}, AUC@Test: {test_score:.4f}')
            loss_models[stage, 1], loss_models[stage, 2] = val_score, test_score

        
        val_auc, te_auc = loss_models[best_stage, 1], loss_models[best_stage, 2]
        print(f'Best validation stage: {best_stage},  AUC@Val: {val_auc:.4f}, final AUC@Test: {te_auc:.4f}')
        y, proba_dnn = tester.test(test_loader)    
        proba_deepforest = Deepforest.predict_proba(X_test)  
        proba_gb = en_proba_score(net_ensemble, test_loader)
        y_proba=0.4*proba_gb+0.3*proba_dnn+0.3*proba_deepforest[:,1]
        result1= cal_metrics(y,y_proba.round(),y_proba,args=args,tm=i, plot=True)
        result.append(result1)

    save_result(result,args)
    cal_time(time.time()-st)
    ti = time.strftime('%m-%d %H-%M-%S',time.localtime(time.time()))
    print('current time:',ti)