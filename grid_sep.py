import os 
import pandas as pd
import numpy as np

# Grids separation
# Listing all files in the directorey
files  = os.listdir()[4:]
# the [4:] slie is the all datafiles in the directory
columns = ["grid_num", "time", "cell_id", "sms_in", "sms_out", "call_in", "call_out", "internet"]
""" 
grid_cell_tn.csv
this is a file generated from previuos expierments has only 2 cols ["grid_num", "cell_id"]
that conatains all grids and cells in the datasets
"""
grid_cell = pd.read_csv("grid_cell_tn.csv")
# getting only the unique grids in that file
grids = grid_cell["grid_num"].unique()
# looping through each grid_num 
for grid in grids:
    #Creating an empty dataframe for the grid num
    grid_df = pd.DataFrame(columns =["grid_num", "time", "cell_id", "rate"])
    #looping through each data file
    for f in files:
        #reading the file
        df = pd.read_csv(f, sep = "\t", header = None)
        # renaming the cols
        df.columns = columns
        # summing the rates
        df["rate"] = df.loc[:, ["sms_in", "sms_out", "call_in", "call_out", "internet"]].sum(axis = 1)
        # dropping the columns of the indinvidual rates after summation
        df.drop(columns = ["sms_in", "sms_out", "call_in", "call_out", "internet"], inplace = True)
        # getting all measurment inside the desired grid num in the file
        df = df.loc[df["grid_num"] == grid]
        #concatination into a new df
        grid_df = pd.concat([grid_df, df], ignore_index = True)
        print("==========={} is extracted for {}============".format(f, grid))
    # finally saving that into its file in a directory called grids
    grid_df.to_csv("grids/{}.csv".format(grid))
