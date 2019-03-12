# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Model, load_model

from sklearn.metrics import mean_squared_error
from numpy import split
from numpy import array

class Predictor(object):

    def __init__(self):
        self.model = self.load_models('./simple_model.h5')

    def log(self, X):
        return np.log10(X + 1.0)
    def unlog(self, X):
        return np.clip(np.power(10., X) - 1.0, 0.0, None)

    def load_models(self, fn):
        print('Load Previous Models')
        model = load_model(fn)
        return model

    def predict(self, name, dataset_raw, figure):
        # evaluate model and get scores
        n_input = 7
        n_output = 14
        cut_day = 28
        mean, std = 1.2474406, 0.26625848

        #dataset_raw = input_df.iloc[0:1,1:]
        dataset = dataset_raw.values
        test_raw = dataset[-cut_day:-n_output]

        # Use the last row of training to predict future n_days
        test = np.reshape(dataset[-cut_day:-cut_day+n_input], (-1, n_input, 1))

        testPred = self.model.predict(test)
        testPred = self.unlog(testPred * std + mean)
        testPred = testPred[0]
        print(testPred, test_raw[-n_output:])

        mse = mean_squared_error(testPred, test_raw[-n_output:])
        print("MSE:", mse)

        start_day = len(dataset) - cut_day

        truePredictPlot = np.empty_like(dataset_raw)
        truePredictPlot[:] = np.nan
        truePredictPlot[start_day:start_day+n_output] = dataset_raw.iloc[start_day:start_day+n_output]


        testPredictPlot = np.empty_like(dataset_raw)
        testPredictPlot[:] = np.nan
        testPredictPlot[start_day:start_day+n_output] = testPred

        # ax = figure.add_subplot(111)
        ax = figure
        ax.plot(dataset_raw.values[:start_day], label="history")
        ax.plot(truePredictPlot, label="real")
        ax.plot(testPredictPlot, label="pred")

        ax.set_title(name)
        ax.set_xlabel("days")
        ax.set_ylabel("num of visit")
        ax.legend()
        # plt.xlim(400, 550)
        # plt.autoscale(axis='y')
        # plt.show()

"""## Simpe LSTM"""



def to_supervised(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0], train.shape[1]))
    X, y = list(), list()

    # step over the entire history one time step at a time
    for row in data:
        in_start = 0
        for _ in range(len(row)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance
            if out_end < len(row):
                x_input = row[in_start:in_end]
                x_input = x_input.reshape((len(x_input), 1))
                X.append(x_input)
                y.append(row[in_end:out_end])
            # move along one time step
            in_start += 1
    return array(X), array(y)

# train the model
def build_model(train, n_input, n_output):
    # prepare data
    train_x, train_y = to_supervised(train, n_input, n_output)

    # define parameters
    verbose, epochs, batch_size = 2, 20, 16

    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    # define model
    model = Sequential()
    model.add(LSTM(200, input_shape=(n_timesteps, n_features), return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

#evaluate_model(train, test, n_input, n_output)

if __name__ == '__main__':
    main()
