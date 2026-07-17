from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import numpy as np

class SVM_class:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def best_fit(self):
        param_grid = {
        'C': np.arange(0.1,1,0.1),
        'gamma': [0.01,0.05,0.1,0.5]
        }
        svm = SVC(kernel='rbf', random_state=42)
        svm_grid_search = RandomizedSearchCV(svm, param_distributions=param_grid, n_iter=20, scoring='neg_log_loss', n_jobs=8, cv=5, verbose=2)
        svm_grid_search.fit(self.X_train, self.y_train)
        best_svm_model = svm_grid_search.best_estimator_
        return {'best': best_svm_model, 'model': svm_grid_search}
    def predict(self, X_test):
        best_model = self.best_fit()['best']
        return best_model.predict(X_test)