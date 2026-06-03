# -CodeAlpha__sales-prediction-using-python
---> DATA CLEANING $ PREPROCESSING:-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Load dataset
df = pd.read_csv('advertising.csv')
print("Shape:", df.shape)
print(df.head())
# Handle missing values
df.dropna(inplace=True)           # Drop rows with nulls
# OR: df.fillna(df.mean(), inplace=True)  # Fill with mean
# Encode categorical columns
le = LabelEncoder()
if 'Platform' in df.columns:
    df['Platform'] = le.fit_transform(df['Platform'])
if 'Segment' in df.columns:
    df['Segment'] = le.fit_transform(df['Segment'])
# Feature & Target split
X = df.drop('Sales', axis=1)
y = df['Sales']
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Preprocessing complete. Features:", X.columns.tolist()
