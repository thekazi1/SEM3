import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy import stats

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

response = requests.get(url)
csv_data = StringIO(response.text)
df = pd.read_csv(csv_data, header=None, names=columns)

print("Before Data Cleaning:")
print(df.info())
print(df.head())

df_imputed = df.copy()

# Handle missing values (numeric)
numeric_cols = df_imputed.select_dtypes(include=np.number).columns
df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(df_imputed[numeric_cols].mean())

# Handle missing values (categorical)
categorical_cols = df_imputed.select_dtypes(include='object').columns
for col in categorical_cols:
    df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)

# Clean column names
df_imputed.columns = df_imputed.columns.str.strip().str.lower().str.replace(' ', '_')

# Detect outliers using Z-score
z_scores = np.abs(stats.zscore(df_imputed.select_dtypes(include=np.number)))
outliers = (z_scores > 3)

# Remove outliers
df_cleaned = df_imputed[~np.any(outliers, axis=1)]

print("\nAfter Data Cleaning:")
print(df_cleaned.info())
print(df_cleaned.head())

print("\nSummary Statistics Before Cleaning:")
print(df.describe())

print("\nSummary Statistics After Cleaning:")
print(df_cleaned.describe())
