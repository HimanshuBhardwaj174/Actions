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
    data = pd.read_csv(loc)
    return data

def processing(df:pd.DataFrame)->pd.DataFrame:
    sel = OrdinalEncoder()
    df['Placed'] = sel.fit_transform(df['Placed'].values.reshape(-1,1))
    return df

def save(path_,train_data,test_data):
    os.makedirs(path_)
    train_data.to_csv(os.path.join(path_,'train_processed.csv'))
    test_data.to_csv(os.path.join(path_,'test_processed.csv'))

def main():
    logger = setup_logging()
    train_data = load_data('./data/interim/train_data.csv')
    test_data = load_data('./data/interim/test_data.csv')
    
    logger.info('Processed Data')
    train_data = processing(train_data)
    test_data = processing(test_data)

    save(os.path.join('./data','processed'),train_data,test_data)



if __name__=='__main__':
    main()