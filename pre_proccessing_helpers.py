# ======= Packages ========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# ======= Functions ========
def check_values_count(data):
    """Use pandas value_counts function for each column in the Dataframe


    Parameters
    ----------

    data: A pandas DataFrame


    Returns
    -------

    None


    """

    for column in data:
        if column == 'test_date':
            continue
        count = data[column].value_counts()
        print(f"the different values for {column}:\n {count} \n")


def map_columns(data, column, dict_costum = False , map_dict = None):
    """Map values for given dataset, column name and dictionary
    

    Parameters
    ----------

    data: A pandas DataFrame
    column: A string for the column name of the column to map
    dict_costum: A bool to determine if the dict is user defined or not
    map_dict: A dictionary supplied by the user (defined the mapping)
    

    Returns
    -------

    None


    """

    if dict_costum:
        data[column] = data[column].map(map_dict)
    else:
        dict_def = {0 : 'no' + ' ' + column, 1 : column}
        data[column] = data[column].map(dict_def)


def plot_dataset(data, plot_object, object_name):
    """Plot the data according to given plot object


    Parameters
    ----------

    data: A pandas DataFrame 
    plot_object: An object for plotting
    object_name: A string for the object's name
    

    Return
    ------
    
    None


    """

    data = data.copy(deep=True)
    fig, axs = plt.subplots(2, 4, figsize = (22.0, 16.0))
    i, j = (0, 0)
    for column in tqdm(data.columns):
        if column == 'test_date' or column == 'corona_result':
            continue
        
        ax = axs[i,j]
    
        if object_name == 'countplot':
            ax = plot_object(x=column, data=data, ax=ax)
            ax.bar_label(ax.containers[0], padding=1)
        if object_name == 'barplot':
            plot_object(x=column, y='corona_result',data=data, ax=ax)
        
        if j == 3:
            i = i + 1
            j = 0
            continue

        j = j + 1

    fig.tight_layout()