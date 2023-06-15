import pandas as pd
# Read the CSV file
df = pd.read_csv('train.csv')

# Replace non-ascii characters
df.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)

# Write to a space-separated text file
df.to_csv('yourfile.txt', sep=' ', index=False, header=False, float_format='%.8f')