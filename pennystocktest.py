# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:09:19 2022

@author: jpv88
"""
import datetime
import matplotlib.pyplot as plt
import pandas as pd

    
df = pd.read_csv ('pennystocks_2021.csv', index_col='ticker')
hd = pd.read_csv('NVDA.csv')
earnings = pd.read_csv('NVDA_earnings.csv')
earnings = earnings['date'].values

# %%

df_dates = list(df.columns.values[1:-1])
hd_dates = list(hd['Date'].values)
for i, date in enumerate(hd_dates):
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    hd_dates[i] = '{0}/{1}/{2:02}'.format(dt.month, dt.day, dt.year % 100)
    
price = list(hd['Adj Close'].values)

first_date_idx = df_dates.index(hd_dates[0])
price = first_date_idx*[price[0]] + price

for i, date in enumerate(df_dates[first_date_idx:]):
    if date not in hd_dates:
        price.insert(i, price[i-1])
        
mentions = df.loc['SNDL'].values[1:-1]
mentions = [(i-min(mentions))/(max(mentions)-min(mentions)) for i in mentions]
price = [(i-min(price))/(max(price)-min(price)) for i in price]
# %%

plt.plot(mentions)
plt.plot(price)
plt.title('SNDL')


# %% 

earnings = list(earnings)

NVDA_price = hd['Close'].values
hd_dates = list(hd['Date'].values)

for i, date in enumerate(hd_dates):
    dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    hd_dates[i] = dt
    
for i, date in enumerate(earnings):
    dt = dt = datetime.datetime.strptime(date, '%m/%d/%Y')
    earnings[i] = dt
    
# %%

earnings_idx = []
for date in earnings:
    earnings_idx.append(hd_dates.index(date))
    
traces = []
for idx in earnings_idx:
    trace = NVDA_price[idx-22:idx+22]
    center = NVDA_price[idx]
    trace = trace/center
    traces.append(trace)


ups = []
for i, trace in enumerate(traces):
    if trace[23] > trace[22]:
        ups.append(i)

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
  