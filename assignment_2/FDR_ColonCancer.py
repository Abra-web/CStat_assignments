import numpy as np
from scipy.stats import bernoulli
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Step 1: Load the data
data = pd.read_csv('colon-labeled.csv')

print(data.columns[-1])

# # Assuming your data has independent variables X and dependent variable y
# X = data.drop('dependent_variable_column_name', axis=1)  # Replace 'dependent_variable_column_name' with actual column name
# y = data['dependent_variable_column_name']  # Replace 'dependent_variable_column_name' with actual column name
#
# # Step 2: Perform logistic regression
# logit_model = sm.Logit(y, sm.add_constant(X))  # Add constant term
# result = logit_model.fit()
#
#
# # Add p-values and corrected p-values to the dataframe
# data['p_values'] = result.pvalues
# data['corrected_p_values'] = corrected_p_values