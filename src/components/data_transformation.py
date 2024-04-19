import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def getData_tranformer_object(self):
        '''
        This function is responsibe for data tranformation
        '''
        try:
            cols = [
                'fixed acidity', 'volatile acidity',
                'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide',
                'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol',
                ]
            
            numeric_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ("scaler", MinMaxScaler())
                ]
            )
            return numeric_pipeline
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_tranformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train & test data completed")
            logging.info("Obtaining tranformation object")

            preprocessing_obj = self.getData_tranformer_object()

            target_column_name = 'quality'

            input_feature_train = train_df.drop(columns=[target_column_name], axis=1)
            le = LabelEncoder()
            target_feature_train = train_df[target_column_name]
            target_feature_train = le.fit_transform(target_feature_train)

            input_feature_test = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test = test_df[target_column_name]
            target_feature_test = le.fit_transform(target_feature_test)



            logging.info(
                "Applying preprocessing object on training and testing dataframe"
            )
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test)
            ]

            logging.info("Saved preprocessing objects")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr, test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)