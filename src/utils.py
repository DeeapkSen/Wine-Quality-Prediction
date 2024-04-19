import os
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.logger import logging
import matplotlib.pyplot as plt

def evalute_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            params = param[list(models.keys())[i]]

            gridsearch = GridSearchCV(model, params, cv=5)
            gridsearch.fit(X_train, y_train)

            model.set_params(**gridsearch.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = model.score(X_train, y_train)
            test_model_score = model.score(X_test, y_test)

            report[list(models.keys())[i]] = test_model_score
            # report.append(test_model_score)

        return report
    except Exception as e:
        raise CustomException(e, sys)
    

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    
def plot_boxplot(data):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.boxplot(data)
    plt.title('Outliers')
    # plt.show()

def target_label(df):
    df['quality'] = df['quality'].replace(3, 5)
    df['quality'] = df['quality'].replace(4, 5)
    df['quality'] = df['quality'].replace(7, 6)
    df['quality'] = df['quality'].replace(8, 6)
    return df

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

