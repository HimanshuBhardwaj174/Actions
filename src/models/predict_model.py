import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from dvclive import Live
import logging
import pickle
import os
import json

def setup_logging():
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the minimum log level

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Set the minimum log level for the console handler

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    return logger

def load_data(loc:str)->pd.DataFrame:
    df = pd.read_csv(loc)
    return df

def save_metrics(dic):
    with open('metrics.json','w') as f:
        json.dump(dic,f)

def main():

    logger = setup_logging()
    with open('models/model.pkl','rb') as f:
        rf = pickle.load(f)

    test_data = load_data('./data/processed/test_processed.csv')
    y_pred = rf.predict(test_data.drop(columns='Placed'))


    dic = {
        'accuracy':accuracy_score(y_true=test_data['Placed'],y_pred=y_pred),
        'precsion':precision_score(y_true=test_data['Placed'],y_pred=y_pred),
        'recall':recall_score(y_true=test_data['Placed'],y_pred=y_pred),
        'f1_score':f1_score(y_true=test_data['Placed'],y_pred=y_pred)
    }

    save_metrics(dic)
    
    with Live(save_dvc_exp=True) as live:
        logger.info('Accuracy')
        live.log_metric('Accuracy',accuracy_score(y_true=test_data['Placed'],y_pred=y_pred))

        logger.info('precision')
        live.log_metric('Precision',precision_score(y_true=test_data['Placed'],y_pred=y_pred))

        logger.info('Recall')
        live.log_metric('Recall',recall_score(y_true=test_data['Placed'],y_pred=y_pred))

        logger.info('f1_score')
        live.log_metric('f1_score',f1_score(y_true=test_data['Placed'],y_pred=y_pred))

    





if __name__=='__main__':
    main()




