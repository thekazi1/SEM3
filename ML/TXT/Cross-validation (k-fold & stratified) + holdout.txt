import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic classification data (same as PDF)
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)

# K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("K-Fold Cross-Validation Scores:", kfold_scores)
print("K-Fold Average Accuracy:", np.mean(kfold_scores))

# Stratified K-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
print("Stratified K-Fold Cross-Validation Scores:", stratified_scores)
print("Stratified K-Fold Average Accuracy:", np.mean(stratified_scores))

# Holdout validation (train/test split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
holdout_accuracy = accuracy_score(y_test, y_pred)
print("Holdout Validation Accuracy:", holdout_accuracy)
