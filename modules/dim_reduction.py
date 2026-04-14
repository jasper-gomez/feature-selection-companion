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
    
def calculate_dim_reduction(df, variance_threshold, mvr_threshold, corr_threshold, target_col):
    var_summary_table, variance_fig = _calculate_variance(df, variance_threshold)
    mvr_summary_table, mvr_fig = _calculate_missing_value_ratio(df, mvr_threshold)
    corr_summary_table, corr_fig = _calculate_correlation(df, target_col, corr_threshold)

    # consolidator that consolidats all features and how many times got flagged
    # as a removable feature across all 3 methods (descending order)
    all_flagged_features = pd.concat([var_summary_table, mvr_summary_table, corr_summary_table])
    feature_counts = all_flagged_features['Feature'].value_counts()
    consolidated_table = feature_counts.to_frame().reset_index()
    consolidated_table.columns = ['Feature', 'Flag Count']
    consolidated_table = consolidated_table.sort_values(by='Flag Count', ascending=False)

    return consolidated_table, var_summary_table, variance_fig, mvr_summary_table, mvr_fig, corr_summary_table, corr_fig

# ======================================================================================
#                                   HELPER FUNCTIONS
# ======================================================================================


# ======================== VARIANCE THRESHOLDING =====================================

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
    plt.axvline(x=variance_threshold, color='red', linestyle='--', label=f'Numeric Threshold ({variance_threshold})')
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

# ======================== MISSING VALUE RATIO =====================================

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
    plt.axvline(x=mvr_threshold, color='red', linestyle='--', label=f'Missing Value Threshold ({mvr_threshold})')
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

# ======================== CORRELATION =====================================

def _calculate_correlation(df, corr_threshold, target_col=None):

    if target_col is None:
        return _corr_no_target(df, corr_threshold)
    else:
        return _corr_with_target(df, target_col, corr_threshold)

def _corr_no_target(df, corr_threshold):

    corr_matrix = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    corr_fig = plt.gcf() # Capture the figure object

    # Identify highly correlated pairs
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_summary_table = upper_tri.stack().reset_index()
    corr_summary_table.columns = ['Feature_1', 'Feature_2', 'Correlation']
    corr_summary_table = corr_summary_table[corr_summary_table['Correlation'].abs() > corr_threshold]
    corr_summary_table = corr_summary_table.sort_values(by='Correlation', key=abs, ascending=False)

    return corr_summary_table, corr_fig

def _corr_with_target(df, target_col, corr_threshold):

    corr_matrix = df.corr()
    target_corr = corr_matrix[target_col].drop(target_col)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=target_corr.values, y=target_corr.index, palette=['red' if abs(x) > corr_threshold else 'blue' for x in target_corr.values])
    plt.axvline(x=corr_threshold, color='red', linestyle='--', label=f'Correlation Threshold ({corr_threshold})')
    plt.axvline(x=-corr_threshold, color='red', linestyle='--')
    plt.title(f'Correlation of Features with Target Variable: {target_col}')
    plt.xlabel('Correlation Coefficient')
    plt.legend(['Threshold'])
    plt.tight_layout()
    
    corr_fig = plt.gcf() # Capture the figure object

    # Identify features with high correlation to target variable
    corr_summary_table = target_corr[abs(target_corr) > corr_threshold].sort_values(key=abs, ascending=False).reset_index()
    corr_summary_table.columns = ['Feature', 'Correlation']

    return corr_summary_table, corr_fig