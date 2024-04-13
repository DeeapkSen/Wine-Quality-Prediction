import os
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from src.logger import logging
import matplotlib.pyplot as plt

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    
# def plot_boxplot(data):
#     plt.figure(figsize=(10, 6))
#     plt.subplot(1, 2, 1)
#     plt.boxplot(data)
#     plt.title('Outliers')
    
#     plt.show()

def outliers(col):
    Q1, Q3 = np.percentile(col, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range

def remove_outlier(df):
    cols = df.select_dtypes(include='float64').columns
    for column in cols:
        lr, ur = outliers(df[column])
        df[column] = np.where(df[column]>ur, ur, df[column])
        df[column] = np.where(df[column]<lr, lr, df[column])
    logging.info("Removed Outlier values")
    plt.show()
    return df

