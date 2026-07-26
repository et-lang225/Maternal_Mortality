from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import cross_val_score 
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval 
from hyperopt.early_stop import no_progress_loss

class SVM_class:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.svm = SVC(kernel='rbf', random_state=42)
        self.space = {
        'C': hp.loguniform('C', np.log(1), np.log(500)),
        'gamma': hp.loguniform('gamma', np.log(0.01), np.log(1))
        }
        self.best_params = None
    def objective(self, params):
        svm = SVC(**params, kernel='rbf', random_state=42)
        scores = cross_val_score(svm, self.X_train, self.y_train, cv=5, scoring="average_precision", n_jobs=-1)
        mean_avgpre = np.mean(scores)
        return {'loss': -mean_avgpre, 'status': STATUS_OK}
    def svm_Kfold(self, max_evals=10):
        trials = Trials()
        best_hyperparams = fmin(fn=self.objective, space=self.space, algo=tpe.suggest, max_evals=max_evals, trials=trials, early_stop_fn=no_progress_loss(30))
        self.best_params = space_eval(self.space, best_hyperparams)
        print(self.best_params)
    def SearchCV(self):
        param_grid = {
        'C': [0.05,0.1,0.2],
        'gamma': [0.001, 0.01,0.1]
        }
        return RandomizedSearchCV(self.svm, param_distributions=param_grid, n_iter=10, scoring='neg_log_loss', n_jobs=8, cv=5, verbose=2)
    def best_fit(self, search_obj):
        search_obj.fit(self.X_train, self.y_train)
        return print(search_obj.best_estimator_)
    def predict(self, X_test, search_obj):
        return search_obj.predict(X_test)