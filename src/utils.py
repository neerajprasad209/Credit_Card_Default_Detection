import sys, os
import pickle
import pymongo
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = recall_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report
    
    except Exception as e:
            logging.info('Exception occured during model training')
            raise CustomException(e,sys)
    
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
    
    
    
    
def export_collection_as_dataframe(db_name, collection_name):
    try:
        mongo_client = pymongo.MongoClient("mongodb+srv://neerajprasad209:neerajprasad209@neerajprasad209.p3s6l5t.mongodb.net/?retryWrites=true&w=majority")
        collection = mongo_client[db_name][collection_name]
        
        # Print debug indormation
        print("Collection string: ", mongo_client)
        print("Database name: ", db_name)
        print("Collection name: ", collection_name)
        
        
        num_samples = collection.count_documents({})
        if num_samples == 0:
            raise ValueError("Cillection is empty")
        
        df = pd.DataFrame(list(collection.find()))
        
        if "_id" in df.columns.tolist():
            df = df.drop(columns=['_id'],axis=1)
        
        df.replace({"na": np.nan},inplace=True)
        
        return df
        
    except Exception as e:
        raise CustomException(e,sys)