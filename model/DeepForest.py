import time
import sys

sys.path.append("..")
import numpy as np
import pandas as pd
import argparse
from sklearn import metrics
from sklearn.metrics import accuracy_score
from deepforest import CascadeForestClassifier

from src.cv_123 import CV, index_fea
from src.cal_time import cal_time
from src.cal_metrics import cal_metrics,save_result

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NR')
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--predictor', type=str, default='forest')
    parser.add_argument('--method', type=str, default='DF')
    parser.add_argument('--cuda', type=str, default=True)
    args = parser.parse_args()
    return args


