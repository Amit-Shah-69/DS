#pract6A

import numpy as np
from scipy import stats
# Generating sample data for multiple groups
np.random.seed(0) # For reproducibility
group1 = np.random.normal(loc=50, scale=10, size=30) # Group 1 data
group2 = np.random.normal(loc=45, scale=12, size=30) # Group 2 data
group3 = np.random.normal(loc=55, scale=8, size=30) # Group 3 data
# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(group1, group2, group3)
# Print results
print("F-Statistic:", f_statistic)
print("P-Value:", p_value)
# Create labels for groups
labels = ['group1'] * len(group1) + ['group2'] * len(group2) + ['group3'] * len(group3)
# Create a DataFrame for the data
df = pd.DataFrame({'data': data, 'group': labels})
# Fit the one-way ANOVA model
model = ols('data ~ C(group)', data=df).fit()
# Perform one-way ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
# Print ANOVA table
print("ANOVA Table:")
print(anova_table)
# Perform Tukey's HSD test for post-hoc analysis
tukey_result = pairwise_tukeyhsd(data, labels)
# Print the summary of the post-hoc test
print("\nTukey's HSD Test:")
print(tukey_result

#pract6B

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# Generating sample data for multiple groups
np.random.seed(0) # For reproducibility
group1 = np.random.normal(loc=50, scale=10, size=30) # Group 1 data
group2 = np.random.normal(loc=45, scale=12, size=30) # Group 2 data
group3 = np.random.normal(loc=55, scale=8, size=30) # Group 3 data
# Combine all groups into one array
data = np.concatenate([group1, group2, group3])
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


