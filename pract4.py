#pract4B

import numpy as np
arr1 = np.array([1, 2, 3, 4, 5])
print("1D Array:")
print(arr1)
arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\n2D Array:")
print(arr2)
print("\nSum of 1D & 2D Array:", np.sum(arr1))

#pract4B

import pandas as pd
data = [['Alice', 25], ['Bob', 30], ['Charlie', 35], ['David', 40]]
df = pd.DataFrame(data, columns=['Name', 'Age'])
print("DataFrame:")
print(df)
df['City'] = ['New York', 'Los Angeles', 'Chicago', 'Houston']
print("\nDataFrame with Added Column:")
print(df)
filtered_df = df[df['Age'] > 30]
print("\nFiltered DataFrame (Age > 30):")
print(filtered_df)
