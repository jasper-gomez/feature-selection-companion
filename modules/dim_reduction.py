import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

# ======================================================================================
#                                   PUBLIC FUNCTIONS
# ======================================================================================

def get_file(file):
    if file is not None:
        df = pd.read_csv(file)

        return df
    else:
        return None
    
def calculate_dim_reduction(df, variance_threshold, mvr_threshold):
    var_summary_table, variance_fig = _calculate_variance(df, variance_threshold)
    mvr_summary_table, mvr_fig = _calculate_missing_value_ratio(df, mvr_threshold)
    correlation_dict = _calculate_correlation(df)

    return var_summary_table, variance_fig, mvr_summary_table, mvr_fig, correlation_dict

# ======================================================================================
#                                   HELPER FUNCTIONS
# ======================================================================================

def _categorical_diversity(column):
    # calculate the entropy of the column
    prob = column.value_counts(normalize=True)
    entropy = -1 * (prob * np.log2(prob)).sum()

    # normalize by max possible entropy for that many categories
    return entropy / np.log2(len(prob))
    
def _calculate_variance(df, variance_threshold):

    # 1. separate numeric to non-numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    categorical_df = df.select_dtypes(exclude=['number'])
    
    # 2. calculate the variance of each column in the numeric dataframe
    scaled_numeric = numeric_df / numeric_df.mean()
    variance_filter = VarianceThreshold(threshold=variance_threshold)
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
            (row['Feature'] in numeric_df.columns and row['Score'] < variance_threshold) or
            (row['Feature'] in categorical_df.columns and row['Score'] < 0.1)
        ) else 'blue', axis=1
    )

    # 5. GENERATE PLOT OBJECT
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='Score', y='Feature', palette=plot_df['Color'].tolist())
    plt.axvline(x=variance_threshold, color='orange', linestyle='--', label=f'Numeric Threshold ({variance_threshold})')
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


def _calculate_missing_value_ratio(df, mvr_threshold):

    # calculate the missing value ratio of each column
    missing_value_ratio = df.isnull().sum() / len(df)

    # get scores of all features for plotting and summary table
    plot_df = pd.DataFrame({
        'Feature': missing_value_ratio.index,
        'Missing_Value_Ratio': missing_value_ratio.values
    })

    # assign color based on threshold
    plot_df['Color'] = plot_df['Missing_Value_Ratio'].apply(lambda x: 'red' if x > mvr_threshold else 'blue')

    # generate plot object
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='Missing_Value_Ratio', y='Feature', palette=plot_df['Color'].tolist())
    plt.axvline(x=mvr_threshold, color='orange', linestyle='--', label=f'Missing Value Threshold ({mvr_threshold})')
    plt.title('Feature Missing Value Ratios')
    plt.xlabel('Missing Value Ratio')
    plt.legend(['Threshold'])
    plt.tight_layout()
    
    mvr_fig = plt.gcf() # Capture the figure object

    # create a summary table of columns with missing value ratio above the threshold
    # in ascending order
    mvr_summary_table = plot_df[plot_df['Color'] == 'red'].copy()
    mvr_summary_table = mvr_summary_table.drop(columns=['Color'])
    mvr_summary_table = mvr_summary_table.sort_values(by='Missing_Value_Ratio', ascending=True)

    return mvr_summary_table, mvr_fig

def _calculate_correlation(df):

    # calculate the correlation of each column with the target variable