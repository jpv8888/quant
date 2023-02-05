# -*- coding: utf-8 -*-
"""
Created on Sun May 22 13:57:55 2022

@author: 17049
"""

import fetch
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from datetime import date, datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



NVDA, SPY = fetch.yahoo(['NVDA', 'SPY'])
earnings = fetch.yahoo('NVDA', data='earnings')

price = list(NVDA['Close'].values)
dates = list(NVDA['Date'].values)
volume = list(NVDA['Volume'].values)
SPY_price = list(SPY['Close'].values)
SPY_dates = list(SPY['Date'].values)

# %%

# standard is 21 (21 trading days in a month on average), so time_steps = 22
# for the day after

def extract_traces(earnings, dates, data, time_steps=22):
    
    today = date.today()
    earnings_idx = []
    for DT in earnings:
        if datetime.strptime(DT, "%Y-%m-%d").date() < today:
            earnings_idx.append(dates.index(DT))
    
    traces = []
    for idx in earnings_idx:
        trace = data[idx-time_steps+2:idx+2]
        center = data[idx-time_steps+2]
        trace = trace/center
        traces.append(trace)
    
    return traces

NVDA_traces = extract_traces(earnings, dates, price)
SPY_traces = extract_traces(earnings, SPY_dates, SPY_price)
volume_traces = extract_traces(earnings, dates, volume)

NVDA_traces = np.array(NVDA_traces)
SPY_traces = np.array(SPY_traces)
volume_traces = np.array(volume_traces)

# for trace in NVDA_traces[0:10,:]:
#     plt.plot(trace)

# plt.vlines(22, 0.4, 2, ls='--')
    
    
# %%

# earnings_idx = []
# for date in earnings[1:]:
#     earnings_idx.append(dates.index(date))
    
# spy_earnings_idx = []
# for date in earnings[1:]:
#     spy_earnings_idx.append(dates.index(date))
    
# traces = []
# for idx in earnings_idx:
#     trace = price[idx-22:idx+22]
#     center = price[idx-22]
#     trace = trace/center
#     traces.append(trace)
    
# spy_traces = []
# for idx in earnings_idx:
#     trace = SPY_price[idx-22:idx+22]
#     center = SPY_price[idx-22]
#     trace = trace/center
#     spy_traces.append(trace)
    
# for trace in traces:
#     plt.plot(trace)

# plt.vlines(22, 0.4, 3.5, ls='--')

# %%




# split into train and test sets
# train_size = int(len(traces_arr) * 0.67)
# test_size = len(traces_arr) - train_size
# train, test = traces_arr[0:train_size,:], traces_arr[train_size:len(traces_arr),:]
# print(len(train), len(test))

def remove_index(traces_list, idx):
    train = []
    test = []
    for traces in traces_list:
        train.append(np.delete(traces, idx, axis=0))
        test.append(traces[idx,:])
    
    return[train, test]

[train, test] = remove_index([NVDA_traces, SPY_traces], 0)

def split_xy(data):
    train, test = data
    
    train_samples = train[0].shape[0]
    time_steps = train[0].shape[1]
    
    reshapedX = []
    reshapedY = []
    
    for feature in train:
        featureX = feature[:,:time_steps-1]
        featureY = feature[:,time_steps-1]
        reshapedX.append(np.reshape(featureX, (train_samples, time_steps-1, 1)))
        reshapedY.append(np.reshape(featureY, (train_samples, 1, 1)))
    
    trainX = np.concatenate(reshapedX, axis=2)
    trainY = np.concatenate(reshapedY, axis=2)
    trainY = trainY[:,:,0]
    
    test_samples = 1
    
    reshapedX = []
    reshapedY = []
    
    for feature in test:
        featureX = feature[:time_steps-1]
        featureY = feature[time_steps-1]
        reshapedX.append(np.reshape(featureX, (test_samples, time_steps-1, 1)))
        reshapedY.append(np.reshape(featureY, (test_samples, 1, 1)))
        
    testX = np.concatenate(reshapedX, axis=2)
    testY = np.concatenate(reshapedY, axis=2)
    testY = testY[:,:,0]
    
    return [trainX, trainY, testX, testY]
    
test2 = split_xy([train, test])
# %%
    
    
NVDA_traces = NVDA_traces[-80:,:]
SPY_traces = SPY_traces[-80:,:]
        
    
        

correct = 0
acc_over_time = []
accs = []
result = []
for j in range(len(NVDA_traces)):
    
    print(j)
    
    [train, test] = remove_index([NVDA_traces, SPY_traces], j)
    
    [trainX, trainY, testX, testY] = split_xy([train, test])
    
    # train = np.delete(traces_arr, j, axis=0)
    # test = traces_arr[j,:]
    # train_spy = np.delete(spy_traces_arr, j, axis=0)
    # test_spy = spy_traces_arr[j,:]
    

    # trainX = train[:,:23]
    # spy_train_x = train_spy[:,:23]
    # trainY = train[:,23]
    # testX = test[:23]
    # testX = np.reshape(testX, (23, 1))
    # testX_spy = test_spy[:23]
    # testX_spy = np.reshape(testX_spy, (23, 1))
    # testX = np.concatenate((testX, testX_spy), axis=1)
    # testY = test[23]
    # testX = np.reshape(testX, (1, 23, 2))
    
    # # reshape input to be [samples, time steps, features]
    # trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    # train_spy_X = np.reshape(spy_train_x, (spy_train_x.shape[0], spy_train_x.shape[1], 1))
    # trainX = np.concatenate((trainX, train_spy_X), axis=2)
    # testX = np.reshape(testX, (1, 23, 1))
    n_features = trainX.shape[2]
    n_time_steps = trainX.shape[1]
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(8, return_sequences=True, input_shape=(n_time_steps, n_features)))
    model.add(LSTM(8))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, callbacks=[callback])
    
    
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
        result.append(1)
    else:
        result.append(0)
    
    print(correct/(j+1))
    acc_over_time.append(correct/(j+1))
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

ci = 1.96*(0.25/91)**(1/2)
    
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
  