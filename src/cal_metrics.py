import time,sys
sys.path.append('..')
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def cal_metrics(y_true, y_pred, y_score,args, tm, plot=False):
    acc = metrics.accuracy_score(y_true, y_pred)  
    pre = metrics.precision_score(y_true, y_pred)  
    rec = metrics.recall_score(y_true, y_pred)  
    f1 = metrics.f1_score(y_true, y_pred)  
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)
    print(f'Precision is :{pre}')
    print(f'Recall is :{rec}')
    print(f'ACC is: {acc}')
    print(f'F1 is: {f1}')
    print(f'AUC is: {auc}')
    print(f'AUPR is :{aupr}')
    ss = (f'Precision: {pre:.4f}\nRecall: {rec:.4f}\nACC: {acc:.4f}\nF1: {f1:.4f}\nAUC: {auc:.4f}\nAUPR: {aupr:.4f}\n\n')
    pt = [tpr,fpr,recall,precision]
    return ss

def save_result(result, args):
    name = args.dataset
    method = args.method
    cv = 'cv' + str(args.cv)
    file1 = f'/data/wangyz/DTI/EnDGG/output/result/{cv}/{method}_{name}_res.txt'
    file2 = f'/data/wangyz/DTI/EnDGG/output/result/average/{method}_{cv}_{name}.txt'
    with open(file1, mode='w') as f:
        for i in result:
            f.write(i)

    pre, rec, acc, f1, auc, aupr = [], [], [], [], [], []
    with open(file1, 'r') as f:
        for line in f.readlines():
            # print(line)
            if "Precision" in line:
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

    pre, rec, acc, f1, auc, aupr = np.array(pre), np.array(rec), np.array(acc), np.array(f1), np.array(
        auc), np.array(aupr)

    with open(file2, mode='a')as f:
        f.write(
            f'{name}\n{"=" * 50}\ntime:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n{args}\n{"=" * 50}\n')

        f.write(f'Precision: {pre.mean():.4f}±{np.std(pre):.4f}\n')
        f.write(f'Recall: {rec.mean():.4f}±{np.std(rec):.4f}\n')
        f.write(f'ACC: {acc.mean():.4f}±{np.std(acc):.4f}\n')
        f.write(f'F1: {f1.mean():.4f}±{np.std(f1):.4f}\n')
        f.write(f'AUC: {auc.mean():.4f}±{np.std(auc):.4f}\n')
        f.write(f'AUPR: {aupr.mean():.4f}±{np.std(aupr):.4f}\n\n')
        f.close()

