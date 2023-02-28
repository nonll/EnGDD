import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

class CV():
    def __init__(self, cv, n_repeats, inc_matrix) -> None:
        self.cv = cv
        self.inc_matrix = inc_matrix
        self.n_repeats = n_repeats
        self.i, self.j = inc_matrix.shape
        self.trains, self.tests = [], []

    def bance(self,index):
        if self.cv==1:  # h
            t = self.inc_matrix.loc[index]
        elif self.cv==2:  # l
            t = self.inc_matrix.loc[:, index]
        elif self.cv==3:  # hl
            inc = self.inc_matrix.stack().reset_index()
            inc = inc.loc[index]

        if self.cv==1 or self.cv==2:
            inc = t.stack().reset_index()
        # Get the index with a value of 1,0 respectively
        s1 = inc[inc.loc[:, 0].values == 1].index
        s0 = inc[inc.loc[:, 0].values == 0].index
        # Randomly select the same index
        s0 = inc.loc[s0, 0].sample(len(s1)).sort_index().index
        # 
        s1 = np.vstack((inc.loc[s1, 'level_0'].values, (inc.loc[s1, 'level_1'].values))).T
        s0 = np.vstack((inc.loc[s0, 'level_0'].values, (inc.loc[s0, 'level_1'].values))).T
        s = np.vstack((s1, s0))
        return s

    def cv_1(self):
        # h
        print('cv1', self.i)
        lens = self.i
        rkf = RepeatedKFold(n_splits=5, n_repeats=self.n_repeats)
        for train_index, test_index in rkf.split(list(range(lens))):
            self.trains.append(self.bance(train_index))
            self.tests.append(self.bance(test_index))
        return self.trains, self.tests

    def cv_2(self):
        # l
        print('cv2', self.j)
        lens = self.j
        rkf = RepeatedKFold(n_splits=5, n_repeats=self.n_repeats)
        for train_index, test_index in rkf.split(list(range(lens))):
            self.trains.append(self.bance(train_index))
            self.tests.append(self.bance(test_index))
        return self.trains, self.tests

    def cv_3(self):
        print('cv3')
        lens = self.i * self.j
        rkf = RepeatedKFold(n_splits=5, n_repeats=self.n_repeats)
        for train_index, test_index in rkf.split(list(range(lens))):
            self.trains.append(self.bance(train_index))
            self.tests.append(self.bance(test_index))
        return self.trains, self.tests

    @classmethod
    def get_cv(cls, cv, n_repeats, inc_matrix):
        if cv == 1:
            trains, tests = cls(cv, n_repeats, inc_matrix).cv_1()
        elif cv == 2:
            trains, tests = cls(cv, n_repeats, inc_matrix).cv_2()
        elif cv == 3:
            trains, tests = cls(cv, n_repeats, inc_matrix).cv_3()       
        return trains, tests

def index_fea(index,col):
    indexs = np.empty(shape=[0, 1])
    for n in index:
        te = n[0] * col + n[1]
        indexs = np.append(indexs, [[te]], axis=0).astype(int)
    return indexs
