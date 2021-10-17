# ====== Packages =======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from tqdm import tqdm

# ---- data manipulators ----
from sklearn.model_selection import StratifiedShuffleSplit

# ---- evaluation ------
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ====== Functions =======

# this function evaluated the model preformance
def model_evaluation(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f"accuracy_score: {accuracy}\n f1_score: {f1}\n recall_score: {recall}\n")
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    disp.plot()
    disp.ax_.set_xticklabels(['negative','positive'])
    disp.ax_.set_yticklabels(['negative','positive'])
    
    
    
