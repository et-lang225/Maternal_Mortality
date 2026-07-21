from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RF_hyperparams:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.rf = RandomForestClassifier(random_state=42)
    def SearchCV(self):
        param_grid = {
        'n_estimators': np.arange(100,1000,100),
        'max_depth': np.arange(5,50,5),
        'min_samples_split': [2,4,6],
        'min_samples_leaf': [1,2,4,6]
        }
        return RandomizedSearchCV(self.rf, param_distributions=param_grid, n_iter=10, scoring='neg_log_loss', n_jobs=8, cv=5, verbose=2)
    def best_fit(self, search_obj):
        search_obj.fit(self.X_train, self.y_train)
        return print(search_obj.best_estimator_)
    def predict(self, X_test, search_obj):
        return search_obj.predict(X_test)