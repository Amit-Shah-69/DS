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
Interpret the results
alpha = 0.05
if p_value < alpha:
print("Reject the null hypothesis: There is a significant difference between the means of at least two groups.")
else:
print("Fail to reject the null hypothesis: There is no significant difference between the means of the groups.")

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
