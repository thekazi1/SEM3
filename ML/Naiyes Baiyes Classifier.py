import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy of Naive Bayes classifier: {accuracy:.2f}')

# Predict a single test sample
test_sample = X_test[0].reshape(1, -1)
predicted_class = nb_classifier.predict(test_sample)

print(f'Predicted class for the test sample: {predicted_class[0]}')
