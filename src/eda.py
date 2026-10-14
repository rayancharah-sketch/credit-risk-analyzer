import pandas as pd

# Load the dataset into a dataframe (df)
df = pd.read_csv('data/credit_risk_dataset.csv')

# Shape the dataframe
print(df.shape)
# Info
df.info()
# Default ratio for loans, 1 is a default
print(df["loan_status"].value_counts(normalize=True))
# Missing values
print(df.isnull().sum())
# Impossible values
print(df.describe())
# Duplicated 
print("Duplicated rows:" + str(df.duplicated().sum()))