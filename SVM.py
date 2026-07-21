from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import numpy as np

class SVM_class:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.svm = SVC(kernel='rbf', random_state=42)
    def SearchCV(self):
        param_grid = {
        'C': [0.05,0.1,0.2],
        'gamma': [0.01,0.05,0.1]
        }
        return RandomizedSearchCV(self.svm, param_distributions=param_grid, n_iter=9, scoring='neg_log_loss', n_jobs=8, cv=5, verbose=2)
    def best_fit(self, search_obj):
        search_obj.fit(self.X_train, self.y_train)
        return print(search_obj.best_estimator_)
    def predict(self, X_test, search_obj):
        return search_obj.predict(X_test)