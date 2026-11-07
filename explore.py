import pandas as pd

# Load the dataset
df = pd.read_csv("Iris.csv")

# Display first 5 records
print("First 5 records:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Display species distribution
print("\nSpecies Distribution:")
print(df["Species"].value_counts())