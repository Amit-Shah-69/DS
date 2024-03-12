#pract7A

import matplotlib.pyplot as plt
import numpy as np
categories = ['Category A', 'Category B', 'Category C', 'Category D']
values1 = [30, 45, 25, 50]
values2 = [20, 35, 40, 30]
def plot_bar_chart():
plt.figure(figsize=(8, 6))
plt.bar(categories, values1, label='Dataset 1', color='blue', alpha=0.7)
plt.bar(categories, values2, label='Dataset 2', color='orange', alpha=0.7, 
bottom=values1)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Comparison of Dataset 1 and Dataset 2')
plt.legend()
plt.show()
def plot_line_chart():
time_periods = np.arange(1, 6)
trend1 = [10, 25, 30, 20, 35]
trend2 = [15, 30, 25, 40, 30]
plt.figure(figsize=(8, 6))
plt.plot(time_periods, trend1, marker='o', label='Trend 1', linestyle='-', 
color='green')
plt.plot(time_periods, trend2, marker='s', label='Trend 2', linestyle='--', 
color='purple')
plt.xlabel('Time Periods')
plt.ylabel('Values')
plt.title('Trends Over Time')
plt.legend()
plt.show() 
def plot_pie_chart():
plt.figure(figsize=(8, 8))
plt.pie(values1, labels=categories, autopct='%1.1f%%', startangle=90, 
colors=['red', 'yellow', 'blue',
'green'])
plt.title('Percentage Distribution of Categories')
plt.show()
plot_bar_chart()
plot_line_chart()
plot_pie_chart() 

#pract7B

import matplotlib.pyplot as plt
import numpy as np
categories = ['Electronics', 'Clothing', 'Books', 'Home Goods']
time_periods = np.arange(1, 6)
sales_data = {
'Electronics': [30, 45, 60, 40, 55],
'Clothing': [20, 35, 25, 30, 40],
'Books': [15, 20, 30, 25, 35],
'Home Goods': [25, 30, 35, 20, 45]
}
def plot_overall_trend():
plt.figure(figsize=(10, 6))
for category in categories:
plt.plot(time_periods, sales_data[category], marker='o', label=category)
plt.xlabel('Time Periods')
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")
plt.scatter(X_test, y_test, color="black", label="Actual data")
plt.plot(X_test, y_pred, color="blue", linewidth=3, label="Regression line")
plt.title("Simple Linear Regression")
plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (y)")
plt.legend()
plt.show()

#pract7C

import numpy as np
categories = ['Electronics', 'Clothing', 'Books', 'Home Goods']
time_periods = np.arange(1, 6)
sales_data = {
'Electronics': [30, 45, 60, 40, 55],
'Clothing': [20, 35, 25, 30, 40],
'Books': [15, 20, 30, 25, 35],
'Home Goods': [25, 30, 35, 20, 45]
}
def calculate_overall_sales():
total_sales = np.sum([sales_data[category] for category in categories], axis=0)
return total_sales
model = LinearRegression()
model.fit(X_train, y_train)
intercept = model.intercept_[0]
slope = model.coef_[0][0]
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Intercept (b0): {intercept}")
print(f"Slope (b1): {slope}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")
plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression line')
plt.title('Simple Linear Regression')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (y)')
plt.legend()
plt.show()
