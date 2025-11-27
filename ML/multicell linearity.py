import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

X = np.random.rand(100, 3)
Y = 3 * X[:, 0] - 2 * X[:, 1] + X[:, 2] + np.random.normal(0, 0.1, 100)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, Y_train)
ridge_pred = ridge_model.predict(X_test)
print("Ridge:", ridge_model.coef_,
      mean_squared_error(Y_test, ridge_pred),
      r2_score(Y_test, ridge_pred))

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, Y_train)
lasso_pred = lasso_model.predict(X_test)
print("Lasso:", lasso_model.coef_,
      mean_squared_error(Y_test, lasso_pred),
      r2_score(Y_test, lasso_pred))

# ElasticNet Regression
elasticnet_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elasticnet_model.fit(X_train, Y_train)
elasticnet_pred = elasticnet_model.predict(X_test)
print("ElasticNet:", elasticnet_model.coef_,
      mean_squared_error(Y_test, elasticnet_pred),
      r2_score(Y_test, elasticnet_pred))
