import pandas as pd

def get_file(file):
    if file is not None:
        df = pd.read_csv(file)
        return df
    else:
        return None
    
def calculate_variance(df):
    
    # calculate the variance of each column

def calculate_mvr(df):

    # calculate the missing value ratio of each column

def calculate_correlation(df):

    # calculate the correlation of each column with the target variable