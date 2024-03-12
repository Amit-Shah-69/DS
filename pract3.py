#pract3A

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
data = {'Feature1': [10, 20, 30, 40],
'Feature2': [0.1, 0.2, 0.3, 0.4]}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
scaler_standard = StandardScaler()
df_standardized = pd.DataFrame(scaler_standard.fit_transform(df), 
columns=df.columns)
scaler_minmax = MinMaxScaler()
df_normalized = pd.DataFrame(scaler_minmax.fit_transform(df), 
columns=df.columns)
print("\nStandardized DataFrame:")
print(df_standardized)
print("\nNormalized DataFrame:")
print(df_normalized)

#pract3B

import pandas as pd
data = {'Color': ['Red', 'Blue', 'Green', 'Red', 'Green'],
'Size': ['Small', 'Large', 'Medium', 'Small', 'Medium']}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
df_encoded = pd.get_dummies(df, columns=['Color', 'Size'])
print("\nEncoded DataFrame:")
print(df_encoded)


