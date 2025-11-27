from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import numpy as np

# Create a synthetic dataset
X,y = make_classification(n_samples=1000,n_features=10,
                            n_informative=5,n_redundant=2,random_state=42)

# Split the data into training and Testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state=42)

# Define the Random Forest Model
model = RandomForestClassifier(random_state=42)

# Grid search hyperparameters
param_grid = {
    'n_estimators':[50,100,200],
    'max_depth': [None,10,20,30],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4]
}

# Grid Search
grid_search = GridSearchCV(estimator = model, param_grid = param_grid,
                            cv = 5, n_jobs = 1, verbose = 2, scoring = 'accuracy')
grid_search.fit(X_train,y_train)

print("Best parameters from Grid Search:",grid_search.best_params_)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Grid Search Model Accuarcy:", accuracy)

# Randomized Search Hyperparameters
param_dist = {
    'n_estimators':[50,100,200,300],
    'max_depth': [None,10,20,30,40],
    'min_samples_split':[2,5,10,15],
    'min_samples_leaf':[1,2,4,6]
}
random_search = RandomizedSearchCV(estimator=model,param_distributions = param_dist,
                                  n_iter=10, cv=5, n_jobs = 1, verbose = 2,
                                  scoring = 'accuracy',random_state = 42)

# Fit the Randomized search
random_search.fit(X_train,y_train)

print("Best parameters from Randomized search:",random_search.best_params_)

# Evaluate the best model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Randomized search Model Accuracy:",accuracy)
