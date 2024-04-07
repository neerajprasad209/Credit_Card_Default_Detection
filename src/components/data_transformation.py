from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys, os
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.utils import save_object

## Data Transformation config

@dataclass
class DataTransformationconfig:
    preprocessor_ob_file_path = os.path.join('artifacts','preprocessor.pkl')
    
    
## Data transformation config


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
        
        def get_data_transformation_object(self):
            
            try:
                logging.info("Data Transformation initiated")
                
                columns=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
                            'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                            'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                            'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
                
                
                
                logging.info('Pipeline Initiated')
                
                
                
                # Numerical Pipeline
                num_pipeline = Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]
                )
                
                preprocessor = ColumnTransformer('num_pipe', num_pipeline, columns)
                
                
                return preprocessor
            
                
                
            except Exception as e:
                raise CustomException(e,sys)


