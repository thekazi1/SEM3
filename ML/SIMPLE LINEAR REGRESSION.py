import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {'X': [1, 2, 3, 4, 5],
        'Y': [2.2, 4.1, 6.3, 8.2, 10.1]}

df = pd.DataFrame(data)

X = df[['X']]
Y = df['Y']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.4, random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)

intercept = model.intercept_
coefficient = model.coef_[0]

Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)

if len(Y_test) > 1:
    r_squared = r2_score(Y_test, Y_pred)
else:
    r_squared = float('nan')

print(f"Intercept: {intercept}")
print(f"Coefficient: {coefficient}")
print(f"MSE: {mse}")
print(f"R-squared: {r_squared}")
