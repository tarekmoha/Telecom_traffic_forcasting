# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 21:09:34 2021

@author: tmaa6
"""
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

df = pd.read_csv("grids/38.csv", usecols = [2,3,4])
df = df.groupby("time").sum()
df.drop(columns = ["cell_id"], inplace = True)
df.reset_index(inplace= True)
def convert(t):
    return dt.datetime.fromtimestamp(t / 1e3) - dt.timedelta(hours = 1)
df["time"] = df["time"].apply(convert)
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")
df.to_csv("grids/38_final.csv")

plt.figure(figsize=(20,8), dpi=80)
plt.plot(df.iloc[:1008,0])