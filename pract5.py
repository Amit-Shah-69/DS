from scipy import stats
group1 = [20, 25, 30, 35, 40]
group2 = [22, 27, 32, 37, 42]
t_statistic, p_value = stats.ttest_ind(group1, group2)
print("T-Statistic:", t_statistic)
print("P-Value:", p_value)
alpha = 0.05
if p_value < alpha:
print("Reject the null hypothesis: There is a significant difference between the 
means of the two groups.")
else:
print("Fail to reject the null hypothesis: There is no significant difference 
between the means of the two groups.")


