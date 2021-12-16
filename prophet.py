# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:35:03 2021

@author: tmaa6
"""

import pandas as pd
import numpy as np
import prophet as pt
import datetime as dt
import os
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot

# Make sure that the data or DS Col is in format YYYY-MM-DD HH:MM:SS
df = pd.read_csv("38_final.csv")
df = df.reset_index()
df = df.loc[:, ["time", "rate"]]
df = df.rename(columns = {"time": "ds", "rate": "y"})
#converting ds to datetime
df["ds"] = pd.to_datetime(df["ds"])
# unction to get the hour of the ds col
def get_hour(datetime_o):
    return datetime_o.hour
#creatig a new col for the hour
df["hour"] = df["ds"].apply(get_hour)

#col to determine the if the hour is o the active time or not(4am, 4pm)
def active_time(t):
    if(t in range(4, 16)):
        return 1
    else:
        return 0
df["active_time"] = df["hour"].apply(active_time)
#drop the hour colum
df.drop(columns = ["hour"], inplace = True)

#dividing the dataset
X_train = df.iloc[:6000, :]
future = df.iloc[6000:6144, [0,2]]

#inititating the model
model = pt.Prophet(changepoint_range=0.9, changepoint_prior_scale=0.001,
                   n_changepoints = 50, weekly_seasonality= 50,
                   daily_seasonality= 100, holidays_prior_scale= 0.1,
                   seasonality_mode = "multiplicative")
# Adding country holidays
model.add_country_holidays(country_name= "IT")
# add monthly seasonality
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Custom sEasonality for times in day accoding to active time

model.add_seasonality(name='daily_active_time', period=1,
                      fourier_order=10, condition_name='active_time')
# fitting the model
model.fit(X_train)
#making forecsting for the future 1-day
forcast = model.predict(future)
#plotting the output and its components
fig = model.plot(forcast)
model.plot_components(forcast)
a = add_changepoints_to_plot(fig.gca(), model, forcast)

# calculate the error:
df_r = pd.DataFrame(columns = ["y", "yhat"])
df_r["y"] = df.iloc[6000:6144, [1]]
df_r.index = np.arange(df_r.shape[0])
df_r["yhat"] = forcast.loc[:, ["yhat"]]
df_r["error"] = df_r["y"] - df_r["yhat"]
nrmse = np.sqrt((df_r["error"]**2).sum() / df_r.shape[0])/df_r["y"].mean()
print(nrmse)
mape = np.absolute(df_r["error"] / df_r["y"]).sum()/ df_r.shape[0]
print(mape)
