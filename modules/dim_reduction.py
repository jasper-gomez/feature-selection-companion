import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

def get_file(file):
    if file is not None:
        df = pd.read_csv(file)

        return df
    else:
        return None
    
def calculate_dim_reduction(df, threshold):
    var_summary_table, variance_fig = _calculate_variance(df, threshold)
    missing_value_ratio_dict = _calculate_missing_value_ratio(df)
    correlation_dict = _calculate_correlation(df)

    return var_summary_table, variance_fig, missing_value_ratio_dict, correlation_dict

def _categorical_diversity(column):
    # calculate the entropy of the column
    prob = column.value_counts(normalize=True)
    entropy = -1 * (prob * np.log2(prob)).sum()

    # normalize by max possible entropy for that many categories
    return entropy / np.log2(len(prob))
    
def _calculate_variance(df, threshold):

    # 1. separate numeric to non-numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    categorical_df = df.select_dtypes(exclude=['number'])
    
    # 2. calculate the variance of each column in the numeric dataframe
    scaled_numeric = numeric_df / numeric_df.mean()
    variance_filter = VarianceThreshold(threshold=threshold)
    variance_filter.fit(scaled_numeric)

    numeric_scores = scaled_numeric.var()

    # 3. calculate the diversity of each column in the categorical datafram
    cat_scores = {}
    for col in categorical_df.columns:
        cat_scores[col] = _categorical_diversity(categorical_df[col])

    # 4. create a unified scoring dataframe for the plot
    all_scores = pd.concat([numeric_scores, pd.Series(cat_scores)])
    plot_df = pd.DataFrame({
        'Feature': all_scores.index,
        'Score': all_scores.values
    })

    # Determine color: Red if below threshold (0.1 for cat, user-defined for num)
    # Note: You can customize this logic if cat/num have different thresholds
    plot_df['Color'] = plot_df.apply(
        lambda row: 'red' if (
            (row['Feature'] in numeric_df.columns and row['Score'] < threshold) or
            (row['Feature'] in categorical_df.columns and row['Score'] < 0.1)
        ) else 'blue', axis=1
    )

    # 5. GENERATE PLOT OBJECT
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='Score', y='Feature', palette=plot_df['Color'].tolist())
    plt.axvline(x=threshold, color='orange', linestyle='--', label=f'Numeric Threshold ({threshold})')
    plt.title('Feature Variance & Diversity Scores')
    plt.xlabel('Normalized Variance / Diversity Score')
    plt.legend(['Threshold'])
    plt.tight_layout()
    
    variance_fig = plt.gcf() # Capture the figure object

    # 6. create the summary table
    var_summary_table = plot_df[plot_df['Color'] == 'red'].copy()
    var_summary_table = var_summary_table.drop(columns=['Color'])
    var_summary_table = var_summary_table.sort_values(by='Score', ascending=True)

    return var_summary_table, variance_fig


def _calculate_missing_value_ratio(vectorized_df):

    # calculate the missing value ratio of each column

def _calculate_correlation(vectorized_df):

    # calculate the correlation of each column with the target variable