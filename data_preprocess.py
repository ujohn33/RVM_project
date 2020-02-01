import pandas as pd
import csv

dataset = pd.read_csv('Ripley_test_prep.csv')

df = pd.DataFrame(dataset, columns = ['x','y','t'])
# Values to find and their replacements
findL = [-1]
replaceL = [0]

# Select column (can be A,B,C,D)
col = 't';

# Find and replace values in the selected column
df[col] = df[col].replace(findL, replaceL)
df.to_csv('Ripley_test.csv', index=False)
