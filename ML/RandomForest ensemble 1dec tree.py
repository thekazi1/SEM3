import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
dt_y_pred = dt_clf.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_y_pred)

rf_clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
rf_clf.fit(X_train, y_train)
rf_y_pred = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)

rf_clf_50 = RandomForestClassifier(n_estimators=50, max_features='sqrt', random_state=42)
rf_clf_50.fit(X_train, y_train)
rf_50_y_pred = rf_clf_50.predict(X_test)
rf_50_accuracy = accuracy_score(y_test, rf_50_y_pred)

print(f"Decision Tree Accuracy: {dt_accuracy:.2f}")
print(f"Random Forest (100 trees) Accuracy: {rf_accuracy:.2f}")
print(f"Random Forest (50 trees) Accuracy: {rf_50_accuracy:.2f}")
