import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = {
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],  # PERFECT multicollinearity
    'Feature3': [1, 3, 2, 4, 3, 5, 6, 7, 8, 9],
    'Target': [3, 7, 5, 9, 11, 15, 17, 21, 23, 27]
}

df = pd.DataFrame(data)

X = df[['Feature1', 'Feature2', 'Feature3']]
Y = df['Target']

vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                   for i in range(X.shape[1])]

print("\nVIF for Features:")
print(vif_data)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

print("\nIntercept:", model.intercept_)
print("Coefficients:", model.coef_)

Y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(Y_test, Y_pred))
print("R-squared:", r2_score(Y_test, Y_pred))
