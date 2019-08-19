# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 08:18:21 2019

@author: DomJJ
"""

import os
import pandas as pd
import numpy as np
import requests
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


"""
url=r'https://info.bossa.pl/pub/ciagle/omega/cgl/few_last.zip'
myfile = requests.get(url,stream = True)
print(myfile.status_code)
print(myfile.headers.get('content-type'))
with open('D:\Kuba/Gain Capital/DownloadTest','wb') as f:
    f.write(myfile.content)
"""
no_of_day_without_quotes=200
n_days=500
percent_return_indicator=10

return_indicator_period=25

#if n_days<no_of_day_without_quotes:
    #no_of_day_without_quotes=n_days

day10=10
day50=50
day100=100

#sample_source_path=r'D:\Kuba\Gain Capital\OmegaInputs\Sample5'

sample_source_path=os.getcwd()

source_path = os.path.join(sample_source_path,"Sample5")

Input_Headers=["Name","Date","Open","High","Low","Close","Volume"]
Input_Use_Cols=["Date","Close"]

#Alternative paths
#All_Inputs_In_The_Folder = glob.glob(source_path + "/*.txt")

all_Inputs_In_The_Folder = [x for x in os.listdir(source_path) if x.endswith(".txt")]

sample_file_path=os.path.join(source_path,all_Inputs_In_The_Folder[0])
merged_prices=pd.read_csv(sample_file_path,usecols=Input_Use_Cols,dtype={"Date":str,"Close":float})
merged_prices.rename(columns={"Date":"Date","Close":all_Inputs_In_The_Folder[0].split(".")[0]},inplace=True)

for inp_file_name in all_Inputs_In_The_Folder[1:]:
    sample_file_path=os.path.join(source_path,inp_file_name)
    single_names_inputs=pd.read_csv(sample_file_path,usecols=Input_Use_Cols,dtype={"Date":str,"Close":float})
    single_names_inputs.rename(columns={"Date":"Date","Close":inp_file_name.split(".")[0]},inplace=True)
    merged_prices=merged_prices.merge(single_names_inputs,left_on='Date',right_on='Date',how='outer')

#sort by date 
merged_prices.sort_values(by=['Date'],ascending=False,inplace=True,na_position='first')

#set Date as an index
merged_prices.set_index('Date',inplace=True)

#drop inputs if no quotes in a given range
cols_with_quotes=merged_prices[:no_of_day_without_quotes].isnull().all(0)
merged_prices=merged_prices.loc[:,~cols_with_quotes]

#fill days without traiding with latest available price
merged_prices.fillna(method='backfill',inplace=True)

#take latest n_days days
merged_prices = merged_prices[:n_days]

#calculate daily log returns (percentage)
daily_log_returns=(100.*np.log(merged_prices/merged_prices.shift(-1))[:-1]).round(2)

#calculate daily simple returns (percentage)
daily_dir_returns=(100.*(merged_prices/merged_prices.shift(-1)-1)[:-1]).round(2)

#calculate daily log returns for given period (percentage)
daily_log_returns_indicator_period=(100.*np.log(merged_prices/merged_prices.shift(-return_indicator_period))[:-1]).round(2)

#10day log returns
n10day_log_returns=(100.*np.log(merged_prices/merged_prices.shift(-day10))[:-1]).round(2)
#50day log returns
n50day_log_returns=(100.*np.log(merged_prices/merged_prices.shift(-day50))[:-1]).round(2)
#100day log returns
n100day_log_returns=(100.*np.log(merged_prices/merged_prices.shift(-day100))[:-1]).round(2)

#derive buy indicator foa a given period True>> 'BUY' label, False >> 'SELL' label
BUY_indicator=daily_log_returns_indicator_period>percent_return_indicator

#print(n10day_log_returns)

#TODO
#descriptive statistics
#graphs?
#outliers

np_BUY_indicator=np.array(BUY_indicator)
np_n10day_log_returns=np.array(n10day_log_returns)

print(np_BUY_indicator.shape)
print(np_n10day_log_returns.shape)

BUY_indicator.to_csv("BuyInd")

###10 Day prediction
max_dimension=np_BUY_indicator.shape[0]-day10
X_train, X_test, Y_train, Y_test = train_test_split(n10day_log_returns[:max_dimension-return_indicator_period], np_BUY_indicator[return_indicator_period:max_dimension], test_size=0.2)
clf=RandomForestClassifier(n_estimators=30)
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
print("Accuracy for",day10,"days: ",metrics.accuracy_score(Y_test, y_pred))

#print(confusion_matrix(Y_test,y_pred))
#print(classification_report(Y_test,y_pred))

###50 Day prediction
max_dimension=np_BUY_indicator.shape[0]-day50
X_train, X_test, Y_train, Y_test = train_test_split(n50day_log_returns[:max_dimension-return_indicator_period], np_BUY_indicator[return_indicator_period:max_dimension], test_size=0.2)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
print("Accuracy for",day50,"days: ",metrics.accuracy_score(Y_test, y_pred))

###100 Day prediction
max_dimension=np_BUY_indicator.shape[0]-day100
X_train, X_test, Y_train, Y_test = train_test_split(n100day_log_returns[:max_dimension-return_indicator_period], np_BUY_indicator[return_indicator_period:max_dimension], test_size=0.3)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
print("Accuracy for",day100,"days: ",metrics.accuracy_score(Y_test, y_pred))
