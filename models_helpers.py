# ====== Packages =======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns
import sklearn as sk
from prettytable import PrettyTable
from tqdm import tqdm

# ---- data manipulators ----
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample

# ---- evaluation ------
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ====== Functions =======
def model_evaluation(y_true, y_pred):
    """Evaluate the ML model according to different metrics


    Parameters
    ----------

    y_true: Data structure containing the true labels of the examples
    y_pred: Data structure containing the prediction of the ML model


    Returns
    -------

    None


    """

    metric_table = PrettyTable()
    metric_table.field_names = ['Metric', 'Score']
    metric_table.add_row(['accuracy', accuracy_score(y_true, y_pred)])
    metric_table.add_row(['recall', recall_score(y_true, y_pred)])
    metric_table.add_row(['presicion', precision_score(y_true, y_pred)])
    metric_table.add_row(['f1 score', f1_score(y_true, y_pred)])
    metric_table.add_row(['f_beta score', fbeta_score(y_true, y_pred, beta=2)]) # recall is more important
    
    print(metric_table)

    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    disp.plot()
    disp.ax_.set_xticklabels(['negative','positive'])
    disp.ax_.set_yticklabels(['negative','positive'])


def data_split(dataset:DataFrame ,n_splits=1, test_size=0.2, train_size=0.8):
    """ split the data into train and test (or train and validation)


    Parameters
    ----------

    dataset: A pandas Dataframe to split
    n_splits: (optional) An integer of number of splits
    test_size: (optional) An inegeger for the propotion of the test set
    train_size: (optional) An inegeger for the propotion of the train set


    Returns
    -------
    
    X_train, y_train, X_test, y_test


    """
    
    spliter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size)
    X = dataset.drop(dataset.columns[-1], axis=1)
    y = dataset[dataset.columns[-1]]
    train_index, test_index = next(spliter.split(X, y))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    return X_train, y_train, X_test, y_test
    

def up_down_sampling(X_train: DataFrame, y_train: DataFrame, ratio: int, up: bool):
    """up/down sample a dataset


    Parameters
    ----------

    X_train: A pandas DataFrame of the the train data without the labels
    y_train: A pandas DataFrame of the labels of the training data
    ratio: An integer for seting the proportion on the minority related to the majority
    up: A bool determine up sampling or down sampling
    

    Returns
    -------
    The upsampled X_train and y_train


    """
    
    X = pd.concat([X_train, y_train], axis=1)
    
    # Separate minority and majority classes (assume minority is labeled as 1)
    negative_samples = X[X[X.columns[-1]]==0]
    positive_samples = X[X[X.columns[-1]]==1]

    # Upsample the minotiry class
    if up:
        upsampled_positive_samples = resample(positive_samples, replace=True, n_samples=int(len(negative_samples)*ratio), random_state=27)
        X = pd.concat([negative_samples, upsampled_positive_samples])
    # Downsample the majority class
    else:
        downsampled_negative_samples = resample(negative_samples, replace=True, n_samples=int(len(positive_samples)*ratio), random_state=27)
        X = pd.concat([positive_samples, downsampled_negative_samples])

    X_train, y_train = X.drop(X.columns[-1], axis=1), X[X.columns[-1]]

    return X_train, y_train

    
        


