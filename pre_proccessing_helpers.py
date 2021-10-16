# ======= Packages ========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ======= Functions ========
# this function check vakues for each column of the dataFrame
def check_values_count(data): 
    for column in data:
        if column == 'test_date':
            continue
        count = data[column].value_counts()
        print(f"the different values for {column}:\n {count} \n")


# this function maps values for given dataset, column name and dictionary
# dict_costum is a boolean: True - dict is given from user else default dict 
def map_columns(data, column, dict_costum = False , map_dict = None):
    if dict_costum:
        data[column] = data[column].map(map_dict)
    else:
        dict_def = {0 : 'no' + ' ' + column, 1 : column}
        data[column] = data[column].map(dict_def)


# this function plots all features according to user define plot object
def plot_dataset(data, plot_object, object_name):
    data = data.copy(deep=True)
    fig, axs = plt.subplots(2, 4, figsize = (22.0, 16.0))
    i, j = (0, 0)
    for column in tqdm(data.columns):
        if column == 'test_date' or column == 'corona_result':
            continue
        
        ax = axs[i,j]
    
        if object_name == 'countplot':
            plot_object(x=column, data=data, ax=ax)
        if object_name == 'barplot':
            plot_object(x=column, y='corona_result',data=data, ax=ax)
        
        if j == 3:
            i = i + 1
            j = 0
            continue

        j = j + 1

    fig.tight_layout()