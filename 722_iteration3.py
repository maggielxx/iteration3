# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 18:40:39 2020

@author: xiaox
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from datetime import datetime
from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

pd.set_option('display.max_columns',20)
pd.set_option('display.max_rows',350)

air_data=pd.read_csv("D:/Uoa/Info722/Assignment/AirPollutionSeoul/Measurement_summary.csv")
print(air_data.head())
print(air_data.tail())
print(air_data.shape)
print(air_data.dtypes)
print(air_data.columns)

print(air_data.describe())
print(air_data.info())
print(air_data.skew())
print(air_data['PM2.5'].value_counts().sort_values(ascending=False))
#air_data['PM2.5'].value_counts().plot(kind='barh')

plt.hist(air_data['PM2.5'])
plt.boxplot(air_data['PM2.5'])
plt.boxplot(air_data['PM10'])
plt.boxplot(air_data['CO'])
plt.boxplot(air_data['SO2'])
plt.boxplot(air_data['NO2'])
plt.boxplot(air_data['O3'])

air_data=air_data[air_data["Station code"]==101]
print(air_data.head())
print(air_data.shape)
print(air_data.describe())

air_data['PM2.5'].quantile(0.998)
air_data['PM2.5'].quantile(0.999)

air_data['PM10'].quantile(0.999)
air_data['PM10'].quantile(0.998)

air_data["PM2.5"] = np.where(air_data["PM2.5"]<200,air_data["PM2.5"],np.nan)
air_data["PM2.5"] = np.where (air_data["PM2.5"]>0,air_data["PM2.5"],np.nan)
air_data["PM10"] = np.where(air_data["PM10"]<200,air_data["PM10"],np.nan)

air_data["PM10"] = np.where (air_data["PM10"]>0,air_data["PM10"],np.nan)
air_data["SO2"] = np.where (air_data["SO2"]>0,air_data["SO2"],np.nan)
air_data["NO2"] = np.where (air_data["NO2"]>0,air_data["NO2"],np.nan)
air_data["O3"] = np.where (air_data["O3"]>0,air_data["O3"],np.nan)
air_data["CO"] = np.where (air_data["CO"]>0,air_data["CO"],np.nan)

plt.boxplot(air_data['PM2.5'])
print(air_data.describe())
print(air_data.info())

air_data["PM2.5"].fillna(air_data["PM2.5"].mean(),inplace=True)
air_data["PM10"].fillna(air_data["PM10"].mean(),inplace=True)
air_data["CO"].fillna(air_data["CO"].mean(),inplace=True)
air_data["SO2"].fillna(air_data["SO2"].mean(),inplace=True)
air_data["NO2"].fillna(air_data["NO2"].mean(),inplace=True)
air_data["O3"].fillna(air_data["O3"].mean(),inplace=True)
print(air_data.describe())
print(air_data.info())

print(air_data.head())
print(air_data.tail())

air_data['averagePM2.5oflast12hours']=''
print(air_data.shape)
print(air_data.iloc[0:12,10])


for num in range(12,25905):
    sum=0;
    for i in range(num-12,num):
        sum=sum+air_data.loc[i].iloc[10]
    air_data.iloc[num,11]=sum/12

    
print(air_data.tail())
#air_data.to_csv("D:/Uoa/Info722/Assignment/CreatedFeature.csv")

#air_data=pd.read_csv("D:/Uoa/Info722/Assignment/CreatedFeature.csv")

status=pd.read_csv("D:/Uoa/Info722/Assignment/status.csv")
print(status.head())
print(status.dtypes)

print(status.shape)
status=status[status["station code"]==101]
print(status.shape)
print(status.head())
print(status.tail())
status.to_csv("D:/Uoa/Info722/Assignment/status1station.csv",index=False)

station_status=pd.read_csv("D:/Uoa/Info722/Assignment/status1station.csv")
print(station_status.head())
print(station_status.tail())

air_data['status']=station_status['status']
print(air_data.head())
print(air_data.tail())
print(air_data.shape)

print(air_data.dtypes)
air_data["Measurement date"] =pd.to_datetime(air_data["Measurement date"])
air_data['averagePM2.5oflast12hours']=pd.to_numeric(air_data['averagePM2.5oflast12hours'])
print(air_data.dtypes)


status_ohe = OneHotEncoder()
X = status_ohe.fit_transform(air_data.status.values.reshape(-1,1)).toarray()
dfOneHot = pd.DataFrame(X, columns = ["Status_"+str(int(i)) for i in range(X.shape[1])])
air_data = pd.concat([air_data, dfOneHot], axis=1)


print(air_data.head())
print(air_data.dtypes)
print(air_data.isnull())  
print(np.isfinite(air_data.all()))
print(air_data.info())


clean_data=air_data.dropna()
clean_data.skew()
print(clean_data.shape)
print(clean_data.isnull())  

print(clean_data.head())
clean_data.info()

print(air_data.corr()["PM2.5"])

columns = clean_data.columns.tolist()
print(clean_data.dtypes)
columns = [c for c in columns if c not in ["Latitude","Longitude","PM2.5","Measurement date","Address","Station code"]]
target = "PM2.5"

train = clean_data.sample(frac=0.8,random_state=1)
test = clean_data.loc[~clean_data.index.isin(train.index)]

print(train.shape)
print(test.shape)

print(columns)
model=LinearRegression()
model.fit(train[columns],train[target])
predictions = model.predict(test[columns])
print(mean_squared_error(predictions,test[target]))
print(mean_absolute_error(predictions,test[target]))
print(np.sqrt(mean_squared_error(predictions,test[target])))
print(r2_score(predictions,test[target]))
print(model.coef_)
print(model.intercept_)

print(clean_data.describe())

model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10,random_state=1)
model.fit(train[columns],train[target])
predictions = model.predict(test[columns])
print(mean_squared_error(predictions,test[target]))
print(mean_absolute_error(predictions,test[target]))
print(np.sqrt(mean_squared_error(predictions,test[target])))
print(r2_score(predictions,test[target]))

model = RandomForestRegressor(n_estimators=150, min_samples_leaf=10,random_state=1)
model.fit(train[columns],train[target])
predictions = model.predict(test[columns])
print(mean_squared_error(predictions,test[target]))
print(mean_absolute_error(predictions,test[target]))
print(np.sqrt(mean_squared_error(predictions,test[target])))
print(r2_score(predictions,test[target]))

