from sklearn.impute import SimpleImputer  ## Handling Missing Values
from sklearn.preprocessing import StandardScaler  #Handling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer  # do different pipeline ko combine ke liye
import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# Data Transformation Config
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

# Data Transformation class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation started')

            # defining which columns should be ordinal encoded and which one should be scaled
            categorical_columns = ['cut','color','clarity']
            numerical_columns = ['carat','depth','table','x','y','z']

            # define the custom ranking for each ordinal variable
            cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories = ['D','E','F','G','H','I','J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline initiated')

            # Numerical Pipeline
            num_pipeline = Pipeline(
                        steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
                ]
            )

            # Categorical 
            cat_pipeline = Pipeline(
                        steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]
            )

            # Column Transformer
            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            logging.info('Error iin data transformation')
            raise CustomException(e,sys)
            


    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completely')
            logging.info(f'Train DataFrame Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing  object')
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            # features into independent and dependent features

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]

            # apply the transformation

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and testing dataset')


            # now concatenate input feature with target
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Preprocessor pickle is created and saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)








