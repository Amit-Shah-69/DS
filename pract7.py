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

