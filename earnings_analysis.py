# -*- coding: utf-8 -*-
"""
Created on Sun May 22 13:57:55 2022

@author: 17049
"""

import fetch
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


NVDA = fetch.yahoo('NVDA')
earnings = fetch.yahoo('NVDA', data='earnings')

price = list(NVDA['Close'].values)
dates = list(NVDA['Date'].values)

# %%

earnings_idx = []
for date in earnings[1:]:
    earnings_idx.append(dates.index(date))
    
traces = []
for idx in earnings_idx:
    trace = price[idx-22:idx+22]
    center = price[idx-22]
    trace = trace/center
    traces.append(trace)
    
for trace in traces:
    plt.plot(trace)

plt.vlines(22, 0.4, 3.5, ls='--')

# %%

traces_arr = np.array(traces)

# split into train and test sets
# train_size = int(len(traces_arr) * 0.67)
# test_size = len(traces_arr) - train_size
# train, test = traces_arr[0:train_size,:], traces_arr[train_size:len(traces_arr),:]
# print(len(train), len(test))

correct = 0

for j in range(len(traces)):
    
    print(j)
    
    train = np.delete(traces_arr, j, axis=0)
    test = traces_arr[j,:]

    trainX = train[:,:23]
    trainY = train[:,23]
    testX = test[:23]
    testY = test[23]
    
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (1, 23, 1))
    
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(23, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    
    
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    
    
    if testY > testX[0,-1,0]:
        dir_test = 1
    else:
        dir_test = -1
    
    
    if testPredict > testX[0,-1,0]:
        dir_predict = 1
    else:
        dir_predict = -1
        
    if dir_test == dir_predict:
        correct += 1
# %%
      

traces_arr = np.array(traces[1:])  
train = traces_arr


trainX = train[:,:23]
trainY = train[:,23]
testX = traces[0]
testY = 178.7

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (1, 23, 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(23, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# %%

ci = 1.96*(0.25/88)**(1/2)
    
# %%
    
dir_test = []
for i, val in enumerate(testY):
    if val > testX[0,-1,0]:
        dir_test.append(1)
    else:
        dir_test.append(-1)

dir_predict = []
for i, val in enumerate(testPredict):
    if val > testX[0,-1,0]:
        dir_predict.append(1)
    else:
        dir_predict.append(-1)


# %%


goods = []
for i, trace in enumerate(traces):
    if (trace[22] > 0.811) and (trace[22] < 0.871):
        goods.append(i)
        
good_traces = [traces[i] for i in goods]
good_traces = [sum(sub_list) / len(sub_list) for sub_list in zip(*good_traces)]
        
plt.plot(good_traces)






 # %%
downs = []
for i, trace in enumerate(traces):
    if trace[23] < trace[22]:
        downs.append(i)



up_traces = [traces[i] for i in ups]
up_traces = [sum(sub_list) / len(sub_list) for sub_list in zip(*up_traces)]

down_traces = [traces[i] for i in downs]
down_traces = [sum(sub_list) / len(sub_list) for sub_list in zip(*down_traces)]
#plt.plot(up_traces)
#plt.plot(down_traces)

weirdos = []
for i, trace in enumerate(traces):
    if trace[0] > 1.2:
        weirdos.append(i)
        
weird_traces = [traces[i] for i in weirdos]
for trace in weird_traces:
    plt.plot(trace)

weird_traces = [sum(sub_list) / len(sub_list) for sub_list in zip(*weird_traces)]
plt.plot(weird_traces)


    
    # %%
traces = [sum(sub_list) / len(sub_list) for sub_list in zip(*traces)]

plt.plot(traces)
plt.vlines(22, 0.95, 1.2, ls='--')
  