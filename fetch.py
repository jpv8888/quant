# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:41:51 2022

@author: 17049
"""

import os
import time

import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By

def yahoo(tickers):
    
    if type(tickers) is not list: tickers = [tickers]
    
    for ticker in tickers:
        if os.path.exists(ticker + '.csv'):
            os.remove(ticker + '.csv')
    
    localdir = os.path.dirname(os.path.realpath(__file__))
    chromeOptions = webdriver.ChromeOptions()
    prefs = { "download.default_directory" : localdir }
    chromeOptions.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(options=chromeOptions)
    
    for ticker in tickers:
      
        link = "https://finance.yahoo.com/quote/" + ticker + "/history?p=" + ticker
        driver.get(link)
        
        t_period = driver.find_element(By.XPATH, '//span[@class="C($linkColor) Fz(14px)"]')
        t_period.click()
        
        max_button = driver.find_element(By.XPATH, '//button[@data-value="MAX"]')
        max_button.click()
        
        download = driver.find_element(By.LINK_TEXT, 'Download')
        download.click()
        
        while not os.path.exists(ticker + '.csv'):
            time.sleep(1)
    
    ticker_data = []
    for ticker in tickers:
        ticker_data.append(pd.read_csv(ticker + '.csv'))
    
    for ticker in tickers:
        os.remove(ticker + '.csv')
        
    return ticker_data
        



   