from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RF_hyperparams:
    def __init__(self):
        pass
    def best_fit(self,X,y):
        param_grid = {
        'n_estimators': np.arange(100,1000,100),
        'max_depth': np.arange(5,50,5),
        'min_samples_split': [2,4,6],
        'min_samples_leaf': [1,2,4,6]
        }
        rf = RandomForestClassifier(random_state=42)
        RF_grid_search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=20, scoring='neg_log_loss', n_jobs=8, cv=5, verbose=2)
        RF_grid_search.fit(X,y)
        best_rf_model = RF_grid_search.best_estimator_
        return {'best': best_rf_model, 'model': RF_grid_search}