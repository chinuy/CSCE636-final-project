# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Model, load_model

from datetime import datetime
from sklearn.metrics import mean_squared_error
from numpy import split
from numpy import array

class Predictor(object):

    def __init__(self):
        self.model = self.load_models('./DNN_model7.h5')

    def log(self, X):
        return np.log10(X + 1.0)
    def unlog(self, X):
        return np.clip(np.power(10., X) - 1.0, 0.0, None)
    def weekday(self, datestr):
        return datetime.strptime(datestr,'%Y-%m-%d').weekday()

    def load_models(self, fn):
        print('Load Previous Models')
        model = load_model(fn)
        return model

    def predict(self, dataset_raw, figure=None):

        test_start = 440 # 2016-09-13

        testPred = self.do_prediction(dataset_raw)
        testPred = testPred[0]

        mae = sum(abs(testPred - dataset_raw.iloc[0,test_start:test_start+63].values)) / 63
        dataset = dataset_raw.iloc[0]
        if figure:
            truePredictPlot = np.empty_like(dataset)
            truePredictPlot[:] = np.nan
            truePredictPlot[test_start: test_start + 63] = dataset.iloc[test_start+1: test_start+64]

            testPredictPlot = np.empty_like(dataset)
            testPredictPlot[:] = np.nan
            testPredictPlot[test_start: test_start+63] = testPred

            ax = figure
            ax.plot(dataset[1:test_start+1], label="history")
            ax.plot(truePredictPlot, label="real")
            ax.plot(testPredictPlot, label="pred")

            ax.set_title(dataset[0])
            ax.set_xticks(np.arange(0, test_start+63, 200))
            # ax.set_xlabel("days")
            ax.set_ylabel("num of visit")
            # ax.set_yscale("log")
            ax.legend(loc='upper left', prop={'size': 8})#, bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=3)

            # plt.xlim(0, test_start+63)
            # plt.autoscale(axis='y')
            # plt.show()
        return mae

    def do_prediction(self, dataset):
        offset = 0.5
        max_size = 366

        all_page = dataset.Page
        train_key = dataset[['Page']].copy()
        train_all = dataset.copy()
        train_all = train_all.iloc[:,1:] * offset

        def get_date_index(date, train_all=train_all):
            for idx, c in enumerate(train_all.columns):
                if date == c:
                    break
            if idx == len(train_all.columns):
                return None
            return idx

        """### Split training dataset into training and testing
        Train data is extract from the original data by the last 181 days
        """

        # Split the dataset into training and testing datasets
        train_end = get_date_index('2016-09-10') + 1
        test_start = get_date_index('2016-09-12')

        train = train_all.iloc[ : , (train_end - max_size) : train_end].copy().astype('float32')
        train = train.iloc[:,::-1].copy().astype('float32')

        """### Feature Engineering"""
        data = [page.split('_') for page in (train_key.Page)]
        access = ['_'.join(page[-2:]) for page in data]
        site = [page[-3][:2] for page in data]
        page = ['_'.join(page[:-3]) for page in data]

        train_key['PageTitle'] = page
        train_key['Site'] = site
        train_key['AccessAgent'] = access

        train_norm = np.log1p(train).astype('float32')

        first_day = 2 # 2017-09-13 is a Wednesday
        test_columns_code = ['w%d_d%d' % ((first_day+i) // 7, (first_day + i) % 7) for i in range(63)]
        test = pd.DataFrame(index=train.index, columns=test_columns_code)

        test.fillna(0, inplace=True)

        test['Page'] = all_page
        test.sort_values(by='Page', inplace=True)
        test.reset_index(drop=True, inplace=True)

        test = test.merge(train_key, how='left', on='Page', copy=False)

        y_cols = test.columns[:63]
        test = test.reset_index()

        train_cols = ['d_%d' % i for i in range(train_norm.shape[1])]

        train_norm.columns = train_cols

        sites = ['zh', 'fr', 'en', 'co', 'ru', 'ww', 'de', 'ja', 'es']
        accesses = ['all-access_spider', 'desktop_all-agents', 'mobile-web_all-agents', 'all-access_all-agents']
        test0 = test.copy()

        y_norm_cols = [c+'_norm' for c in y_cols]
        y_pred_cols = [c+'_pred' for c in y_cols]

        train_norm_diff = train_norm - train_norm.shift(-1, axis=1)
        train_norm_diff.head()

        train_norm_diff7 = train_norm - train_norm.shift(-7, axis=1)
        train_norm_diff7.head()

        train_norm = train_norm.iloc[:,::-1]
        train_norm_diff7m = train_norm - train_norm.rolling(window=7, axis=1).median()
        train_norm = train_norm.iloc[:,::-1]
        train_norm_diff7m = train_norm_diff7m.iloc[:,::-1]

        # load allVisits information
        allVisits = pd.read_csv("meta.csv")
        allVisits = allVisits.sort_values(by='Page')

        def add_median(test, train, train_diff, train_diff7, train_diff7m,
                       train_key, periods, max_periods):
            train =  train.iloc[:,:7*max_periods]

            test = test.merge(allVisits, how='left', on='Page', copy=False)
            test.AllVisits = test.AllVisits.fillna(0).astype('float32')

            for site in sites:
                test[site] = (1 * (test.Site == site)).astype('float32')

            for access in accesses:
                test[access] = (1 * (test.AccessAgent == access)).astype('float32')

            for (w1, w2) in periods:

                df = train_key[['Page']].copy()

                c = 'median_%d_%d' % (w1, w2)
                cm = 'mean_%d_%d' % (w1, w2)
                cmax = 'max_%d_%d' % (w1, w2)

                cd = 'median_diff_%d_%d' % (w1, w2)
                cd7 = 'median_diff7_%d_%d' % (w1, w2)
                cd7m = 'median_diff7m_%d_%d' % (w1, w2)
                cd7mm = 'mean_diff7m_%d_%d' % (w1, w2)

                df[c] = train.iloc[:,7*w1:7*w2].median(axis=1, skipna=True)
                df[cm] = train.iloc[:,7*w1:7*w2].mean(axis=1, skipna=True)
                df[cmax] = train.iloc[:,7*w1:7*w2].max(axis=1, skipna=True)
                df[cd] = train_diff.iloc[:,7*w1:7*w2].median(axis=1, skipna=True)
                df[cd7] = train_diff7.iloc[:,7*w1:7*w2].median(axis=1, skipna=True)
                df[cd7m] = train_diff7m.iloc[:,7*w1:7*w2].median(axis=1, skipna=True)
                df[cd7mm] = train_diff7m.iloc[:,7*w1:7*w2].mean(axis=1, skipna=True)

                test = test.merge(df, how='left', on='Page', copy=False)
                test[c] = (test[c] - test.AllVisits).fillna(0).astype('float32')
                test[cm] = (test[cm] - test.AllVisits).fillna(0).astype('float32')
                test[cmax] = (test[cmax] - test.AllVisits).fillna(0).astype('float32')
                test[cd] = (test[cd] ).fillna(0).astype('float32')
                test[cd7] = (test[cd7] ).fillna(0).astype('float32')
                test[cd7m] = (test[cd7m] ).fillna(0).astype('float32')
                test[cd7mm] = (test[cd7mm] ).fillna(0).astype('float32')

            for c_norm, c in zip(y_norm_cols, y_cols):
                test[c_norm] = (np.log1p(test[c]) - test.AllVisits).astype('float32')

            gc.collect()

            return test

        max_periods = 52
        periods = [(0,1), (1,2), (2,3), (3,4),
                   (4,5), (5,6), (6,7), (7,8),
                   (0,2), (2,4),(4,6),(6,8),
                   (0,4),(4,8),(8,12),(12,16),
                   (0,8), (8,16), (0,12),
                   (0,16), (0, 20), (0, 24), (0, 36), (0, 52),
                  ]

        site_cols = list(sites)
        access_cols = list(accesses)

        for c in y_pred_cols:
            test[c] = np.NaN

        test1 = add_median(test0, train_norm, train_norm_diff, train_norm_diff7, train_norm_diff7m,
                           train_key, periods, max_periods)

        """## Models
        """
        gc.collect()

        """### DNN model"""

        num_cols = (['median_%d_%d' % (w1,w2) for (w1,w2) in periods])
        num_cols.extend(['mean_%d_%d' % (w1,w2) for (w1,w2) in periods])
        num_cols.extend(['max_%d_%d' % (w1,w2) for (w1,w2) in periods])
        num_cols.extend(['median_diff_%d_%d' % (w1,w2) for (w1,w2) in periods])
        num_cols.extend(['median_diff7m_%d_%d' % (w1,w2) for (w1,w2) in periods])
        num_cols.extend(['mean_diff7m_%d_%d' % (w1,w2) for (w1,w2) in periods])

        test2 = test1
        Xm, Xs, Xa, y = test2[num_cols].values, test2[site_cols].values, test2[access_cols].values, test2[y_norm_cols].values

        """## Generate output for Kaggle

        ### Prediction
        """

        Xm_train, Xs_train, Xa_train, y_train = test2[num_cols].values, test2[site_cols].values, test2[access_cols].values, test2[y_norm_cols].values

        predY = self.model.predict([Xm_train, Xs_train, Xa_train])
        predY = predY + test2.AllVisits.values.reshape((-1,1))
        predY = np.expm1(predY)
        predY[predY < 0.5 * offset] = 0

        predY = (predY/offset).round()
        return predY

if __name__ == '__main__':
    df = pd.read_csv('demo.csv').fillna(0)
    target = df.iloc[0:1,:]
    pred = Predictor()
    figure = plt.Figure(figsize=(6,4), dpi=100)
    ax = figure.add_subplot(111)
    mae = pred.predict(target, ax)
    print(mae)

