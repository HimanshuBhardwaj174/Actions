import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from dvclive import Live
import logging

import os

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

def save(path_,train_data,test_data):
    os.makedirs(path_)
    train_data.to_csv(os.path.join(path_,'train_data.csv'))
    test_data.to_csv(os.path.join(path_,'test_data.csv'))


def main():
    logger = setup_logging()
    logger.info('Data Ingestion')
    df = load_data('./data/external/student_placement_large.csv')

    X_train,X_test = train_test_split(df)
    logger.info('Saved Data')
    save(os.path.join('./data','interim'),X_train,X_test)



if __name__=='__main__':
    main()