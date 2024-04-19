import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evalute_model
from dataclasses import dataclass
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

@dataclass
class ModelTrainerConfig:
    model_trainer_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Split train & test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            model = {
                'Decision Tree' : DecisionTreeClassifier(),
                'Logistic Regression' : LogisticRegression(),
                'K-Neighbors' : KNeighborsClassifier(),
                'SVC' : SVC(),
                'Random Forest' : RandomForestClassifier(),
                'XGBClassifier' : XGBClassifier(),
                'ExtraTreesClassifier' : ExtraTreesClassifier(),
                'AdaBoost' : AdaBoostClassifier(),
                'ElasticNet' : ElasticNet(),
            }
            params = {
                'SVC':{
                    'C' : [1, 10, 20],
                    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
                    },
                'Random Forest':{
                        'n_estimators' : [1,5,10]
                    },
                'Logistic Regression':{
                    'C' : [1,5, 10],
                    'solver' : ['liblinear', 'newton-cg', 'newton-cholesky','sag', 'saga']
                    },
                'Decision Tree' :{
                    'criterion' : ['gini', 'entropy', 'log_loss'],
                    'splitter' : ['best', 'random']
                },
                'K-Neighbors' :{
                    'n_neighbors' : [1, 2, 3, 5, 11],
                    'weights' : ['uniform', 'distance'],
                    'algorithm' : ['auto','ball_tree', 'kd_tree', 'brute']
                    },
                'ElasticNet': {
                    'alpha' : [0.2, 0.4, 0.6, 0.8, 0.9],
                    'l1_ratio' : [0.5]
                    },
                'ExtraTreesClassifier': {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'n_estimators' : [8, 16, 32, 64, 128, 256]
                    },
                'AdaBoost' :{
                    'learning_rate' : [.1, .01, .05, .001],
                    'n_estimators' : [8, 16, 32, 64, 128, 256]
                    },
                'XGBClassifier' : {
                    'learning_rate' : [.1, .01, .05, .001],
                    'n_estimators' : [8, 16, 32, 64, 128, 256]
                }
            }
            model_report : dict=evalute_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                              models=model, param=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = model[best_model_name]

            if best_model_score < 0.6:
                print('No best model found')
                # raise CustomException("No best model found", sys)
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.model_trainer_path,
                obj= best_model
            )

            predicted = best_model.predict(X_test)
            scores = best_model.score(X_test, y_test)

            return scores, best_model_name

        except Exception as e:
            raise CustomException(e, sys)




