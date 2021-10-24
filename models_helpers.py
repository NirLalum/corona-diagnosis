# ====== Packages =======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from prettytable import PrettyTable
from tqdm import tqdm

# ---- data manipulators ----
from sklearn.model_selection import StratifiedShuffleSplit

# ---- evaluation ------
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ====== Functions =======

# this function evaluated the model preformance
def model_evaluation(y_true, y_pred):
    metric_table = PrettyTable()
    metric_table.field_names = ['Metric', 'Score']
    metric_table.add_row(['accuracy', accuracy_score(y_true, y_pred)])
    metric_table.add_row(['recall', recall_score(y_true, y_pred)])
    metric_table.add_row(['f1 score', f1_score(y_true, y_pred)])
    metric_table.add_row(['presicion', precision_score(y_true, y_pred)])
    print(metric_table)

    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    disp.plot()
    disp.ax_.set_xticklabels(['negative','positive'])
    disp.ax_.set_yticklabels(['negative','positive'])
    
    
    
