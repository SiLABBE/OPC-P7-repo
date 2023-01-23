import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_matrix(df: pd.DataFrame):
    """
    A function to calculate and plot
    correlation matrix of a DataFrame.
    """
    # Create the matrix
    matrix = df.corr()
    
    # Create cmap
    cmap = sns.diverging_palette(250, 15, s=75, l=40,
                             n=9, center="light", as_cmap=True)
    # Create a mask
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    
    # Make figsize bigger
    fig, ax = plt.subplots(figsize=(16,12))
    
    # Plot the matrix
    _ = sns.heatmap(matrix, mask=mask, center=0, annot=True,
             fmt='.2f', square=True, cmap=cmap, ax=ax)


def KBest_filter(df, k_value=20):
    """
    A function to filter features
    of a DataFrame based on ANOVA F-value.
    """

    from sklearn.feature_selection import SelectKBest, f_classif 

    df_columns = df.columns.tolist()
    X_columns = [col for col in df_columns if col not in ['TARGET','SK_ID_CURR']]
    x = df[X_columns]
    y = df["TARGET"]

    selector = SelectKBest(f_classif, k=k_value)
    selector.fit(x, y)
    selector_mask = selector.get_support()
    filtered_columns = np.array(X_columns)[np.array(selector_mask)].tolist()
    df_filtered = df[['TARGET', 'SK_ID_CURR'] + filtered_columns]
    
    return df_filtered