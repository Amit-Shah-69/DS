#pract2A

import pandas as pd
csv_file_path = 'C:/Users/Mr Danish/Downloads/username.csv'
df_csv = pd.read_csv(csv_file_path)
print("CSV DataFrame:")
print(df_csv)
json_file_path = 'C:/Users/Mr Danish/Downloads/sample1.json'
df_json = pd.read_json(json_file_path)
print("\nJSON DataFrame:")
print(df_json)

#pract2B

import pandas as pd
csv_file_path = 'C:/Users/Mr Danish/Downloads/username.csv'
json_file_path = 'C:/Users/Mr Danish/Downloads/sample1.json'
df_csv = pd.read_csv(csv_file_path)
df_json = pd.read_json(json_file_path)
print("Missing values in CSV DataFrame:")
print(df_csv.isnull().sum())
print("\nMissing values in JSON DataFrame:")
print(df_json.isnull().sum())
df_csv_cleaned = df_csv.dropna()
fill_value = 'Unknown' 
df_json_cleaned = df_json.fillna(fill_value)
numeric_columns_csv = df_csv.select_dtypes(include='number').columns
df_csv_clipped = df_csv.copy()
df_csv_clipped[numeric_columns_csv] = 
df_csv_clipped[numeric_columns_csv].clip(lower=df_csv_clipped[numeric_columns
_csv].mean() - 3 * df_csv_clipped[numeric_columns_csv].std(), 
upper=df_csv_clipped[numeric_columns_csv].mean() + 3 * 
df_csv_clipped[numeric_columns_csv].std(), axis=0)
numeric_columns_json = df_json.select_dtypes(include='number').columns
df_json_clipped = df_json.copy()
df_json_clipped[numeric_columns_json] = 
df_json_clipped[numeric_columns_json].clip(lower=df_json_clipped[numeric_colum
ns_json].mean() - 3 * df_json_clipped[numeric_columns_json].std(), 
upper=df_json_clipped[numeric_columns_json].mean() + 3 * 
df_json_clipped[numeric_columns_json].std(), axis=0)
print("\nCleaned CSV DataFrame:")
print(df_csv_cleaned)
print("\nCleaned JSON DataFrame:")
print(df_json_cleaned)
print("\nClipped CSV DataFrame:")
print(df_csv_clipped)
print("\nClipped JSON DataFrame:")
print(df_json_clipped)

#pract2C

import pandas as pd
data = {'Name': ['John', 'Jane', 'Alice', 'Bob'],
'Age': [25, 30, 22, 35],
'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']}
df_csv = pd.DataFrame(data)
filtered_data = df_csv[df_csv['Age'] > 25]
sorted_data = df_csv.sort_values(by='Age', ascending=False)
grouped_data = df_csv.groupby('City')['Age'].mean().reset_index()
print("Original DataFrame:")
print(df_csv)
print("\nFiltered DataFrame (Age > 25):")
print(filtered_data)
print("\nSorted DataFrame (by Age in descending order):")
print(sorted_data)
print("\nGrouped DataFrame (mean age by City):")
print(grouped_data)
