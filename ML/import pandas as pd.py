import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import urllib.request
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
dataset_file = "iris.data"

if not os.path.exists(dataset_file):
    urllib.request.urlretrieve(url, dataset_file)

columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
data = pd.read_csv(dataset_file, header=None, names=columns)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

correct_predictions = np.sum(y_test == y_pred)
incorrect_predictions = np.sum(y_test != y_pred)

print(f"\nCorrect Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {incorrect_predictions}")

results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print("\nSample Results:")
print(results.head())
