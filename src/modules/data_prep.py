import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def handle_outliers(df, columns, method='cap', multiplier=1.5):
    outlier_count = 0
    df_cleaned = df.copy()

    for column in columns:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outliers = df_cleaned[(df_cleaned[column] < lower_bound) | (df_cleaned[column] > upper_bound)]

        if method == 'cap':
            df_cleaned[column] = np.where(df_cleaned[column] > upper_bound, upper_bound, df_cleaned[column])
            df_cleaned[column] = np.where(df_cleaned[column] < lower_bound, lower_bound, df_cleaned[column])
        elif method == 'remove':
            df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]

        outlier_count += len(outliers)

    return df_cleaned, outlier_count

df_cleaned, outlier_count = handle_outliers(df, numerical_columns, multiplier=2.5)

print(f"Number of outliers identified and capped: {outlier_count}")



