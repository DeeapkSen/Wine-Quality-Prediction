import sys
import os
import numpy as np
import pandas as pd
import warnings

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.utils import remove_outlier, target_label
from src.components.model_train import ModelTrainer, ModelTrainerConfig

warnings.filterwarnings('ignore')


@dataclass
class DataIngestionConfig:
    train_data_path: str= os.path.join('artifacts', 'train.csv')
    test_data_path: str= os.path.join('artifacts', 'test.csv')
    raw_data_path: str= os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or components")

        try:
            df = pd.read_csv("Notebook\Data\winequality-red.csv")
            logging.info("Read the dataset as dataframe")

            target = target_label(df) 

            df = remove_outlier(target)
            # plot_boxplot(df)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train Test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=40)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":

    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_tranformation(train_data, test_data)

    modeltrainer = modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))