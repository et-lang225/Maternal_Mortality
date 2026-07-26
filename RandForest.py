from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score 
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval 
from hyperopt.early_stop import no_progress_loss

class RF_hyperparams:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.rf = RandomForestClassifier(random_state=42)
        self.space = {
        'n_estimators': hp.uniform('n_estimators', 100, 1000),
        'max_depth': hp.choice("max_depth", np.arange(1,50,1,dtype=int)),
        'min_samples_split': hp.choice("min_samples_split", np.arange(2,6,1,dtype=int)),
        'min_samples_leaf': hp.choice("min_samples_leaf", np.arange(2,6,1,dtype=int))
        }
    def objective(self, params):
        rf = RandomForestClassifier(**params, random_state=42)
        scores = cross_val_score(rf, self.X_train, self.y_train, cv=5, scoring="average_precision", n_jobs=-1)
        mean_avgpre = np.mean(scores)
        return {'loss': -mean_avgpre, 'status': STATUS_OK}
    def rf_Kfold(self, max_evals=10):
        trials = Trials()
        best_hyperparams = fmin(fn=self.objective, space=self.space, algo=tpe.suggest, max_evals=max_evals, trials=trials, early_stop_fn=no_progress_loss(30))
        self.best_params = space_eval(self.space, best_hyperparams)
        print(self.best_params)
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