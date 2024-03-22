import numpy as np
from scipy.stats import bernoulli
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler



# Step 1: Load the data
data = pd.read_csv('colon - labled.csv')

data = data.drop(data.columns[0], axis=1)

data["Class"] = data['Class'].map({'Normal': 0, 'Abnormal': 1})



X = data.drop('Class', axis=1)  # Replace 'dependent_variable_column_name' with actual column name
y = data['Class']  # Replace 'dependent_variable_column_name' with actual column name


# Initialize and fit the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X, y)

# Number of bootstrap iterations
n_iterations = 100

# Array to store bootstrap coefficients
bootstrap_coefs = []


# Bootstrap loop
for i in range(n_iterations):
    # Resample the training data
    X_boot, y_boot = resample(X, y, random_state=i)

    # Fit logistic regression model on bootstrapped data
    boot_model = LogisticRegression()
    boot_model.fit(X_boot, y_boot)



    # Store coefficients
    bootstrap_coefs.append(boot_model.coef_[0])


# Calculate p-values
original_coefs = logistic_model.coef_[0]
p_values = []
for j in range(len(original_coefs)):
    coef_values = [coef[j] for coef in bootstrap_coefs]
    p_value = (np.sum(np.abs(coef_values) >= np.abs(original_coefs[j])) + 1) / (n_iterations + 1)
    p_values.append(p_value)

# Create DataFrame with coefficients and p-values
coef_names = X.columns
p_values_df = pd.DataFrame({'Coefficient': coef_names, 'P-value': p_values})

#CHANGE THIS TO CRT FIRST AND THEN BHq - plug p_values into CRT and insert crt pvalues into this line below
p_values_df['corrected_p_values'] = stats.false_discovery_control(p_values)


import seaborn as sns

sns.histplot(data=p_values_df, x='P-value', kde=True,
             bins=20).set_title('Hisgoragram of pvalues before BHq')

plt.show()

sns.histplot(data=p_values_df, x='corrected_p_values', kde=True,
             bins=20).set_title('Hisgoragram of pvalues after BHq')

plt.show()




