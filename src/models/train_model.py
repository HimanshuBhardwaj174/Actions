import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from dvclive import Live
import logging
import pickle
import os
import yaml
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

def model_save(loc,model):
    with open(os.path.join(loc,'model.pkl'),'wb') as f:
        pickle.dump(model,f)

def param_load(loc):
    with open(loc,'r') as f:
        val = yaml.safe_load(f)['model_training']['n_estimator']
    return val
def main():
    logger = setup_logging()
    train_data = load_data('./data/processed/train_processed.csv')
    

    n_estimator = param_load('parmas.yaml')
    rf =  RandomForestClassifier(n_estimators=n_estimator)

    rf.fit(train_data.drop(columns='Placed'),train_data['Placed'])

    logger.info('Saved Model')
    model_save('models',rf)





if __name__=='__main__':
    main()