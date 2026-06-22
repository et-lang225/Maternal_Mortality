from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import numpy as np

class SVM_class:
    def __init__(self):
        pass
    def best_fit(self,X,y):
        param_grid = {
        'C': np.arange(0.1,1,0.1),
        'gamma': [0.01,0.05,0.1,0.5]
        }
        svm = SVC(kernel='rbf', random_state=42)
        svm_grid_search = RandomizedSearchCV(svm, param_distributions=param_grid, n_iter=20, scoring='neg_log_loss', n_jobs=8, cv=5, verbose=2)
        svm_grid_search.fit(X,y)
        best_svm_model = svm_grid_search.best_estimator_
        return {'best': best_svm_model, 'model': svm_grid_search}