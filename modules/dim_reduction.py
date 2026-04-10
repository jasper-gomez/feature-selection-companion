import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def get_file(file):
    if file is not None:
        df = pd.read_csv(file)

        return df
    else:
        return None
    
def calculate_dim_reduction(df, threshold):
    variance_dict_reduced = _calculate_variance(df, threshold)
    missing_value_ratio_dict = _calculate_missing_value_ratio(df)
    correlation_dict = _calculate_correlation(df)

    return variance_dict_reduced, missing_value_ratio_dict, correlation_dict

def _categorical_diversity(column):
    # calculate the entropy of the column
    prob = column.value_counts(normalize=True)
    entropy = -1 * (prob * np.log2(prob)).sum()

    # normalize by max possible entropy for that many categories
    return entropy / np.log2(len(prob))
    
def _calculate_variance(df, threshold):

    # separate numeric to non-numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    categorical_df = df.select_dtypes(exclude=['number'])
    
    # calculate the variance of each column
    variance_filter = VarianceThreshold(threshold=threshold)
    variance_filter.fit(numeric_df / numeric_df.mean())

    # make a mask for the columns that have variance above the threshold
    mask = variance_filter.get_support()

    # get the columns to drop
    numeric_columns_to_drop = df.columns[~mask]

    categorical_columns_to_drop = []
    for col in categorical_df.columns:
        if _categorical_diversity(categorical_df[col]) < 0.1:
            categorical_columns_to_drop.append(col)

    # put in a dictionary the columns to drop and their variance
    variance_dict_reduced = {}
    for col in numeric_columns_to_drop:
        variance_dict_reduced[col] = df[col].var()

    return variance_dict_reduced


def _calculate_missing_value_ratio(vectorized_df):

    # calculate the missing value ratio of each column

def _calculate_correlation(vectorized_df):

    # calculate the correlation of each column with the target variable