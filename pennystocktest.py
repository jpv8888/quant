# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:09:19 2022

@author: jpv88
"""
import datetime
import matplotlib.pyplot as plt
import pandas as pd

    
df = pd.read_csv ('pennystocks_2021.csv', index_col='ticker')
hd = pd.read_csv('SNDL.csv')

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
