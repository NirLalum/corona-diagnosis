import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.categorical import countplot
from tqdm import tqdm

def plot_dataset(data, plot_object, object_name):
    fig, axs = plt.subplots(2, 4, figsize = (22.0, 16.0))
    i, j = (0, 0)
    for column in tqdm(data.columns):
        if column == 'test_date' or column == 'corona_result':
            continue
        
        ax = axs[i,j]
        
        if object_name == 'countplot':
            plot_object(x=column, data=data, ax=ax)
        if object_name == 'barplot':
            plot_object(x=column, data=data, ax=ax)
        
        if j == 3:
            i = i + 1
            j = 0
            continue
        
        j = j + 1

    fig.tight_layout()