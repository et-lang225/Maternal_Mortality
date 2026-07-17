import xgboost as xgb
from sklearn.model_selection import cross_val_score 
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval 
from hyperopt.early_stop import no_progress_loss
import numpy as np

class XGBclass:
    def __init__(self, X, y):
        self.dtrain = xgb.DMatrix(X, y, enable_categorical = True)
        self.space = {
            'max_depth': hp.choice("max_depth", np.arange(1,20,1,dtype=int)),
            'eta'      : hp.uniform("eta", 0, 1),
            'gamma'    : hp.uniform("gamma", 0, 10e1),
            'reg_alpha': hp.uniform("reg_alpha", 10e-7, 10),
            'reg_lambda' : hp.uniform("reg_lambda", 0,1),
            'colsample_bytree': hp.uniform("colsample_bytree", 0.5,1),
            'colsample_bynode': hp.uniform("colsample_bynode", 0.5,1), 
            'colsample_bylevel': hp.uniform("colsample_bylevel", 0.5,1),
            'min_child_weight' : hp.choice("min_child_weight", np.arange(1,10,1,dtype='int')),
            'max_delta_step' : hp.choice("max_delta_step", np.arange(1,10,1,dtype='int')),
            'subsample' : hp.uniform("subsample",0.5,1),
            'objective' : 'binary:logistic',
            'eval_metric' : 'aucpr',
            'seed' : 44
        }
    def xgb_obj(self, space_params, k=5):
        results = xgb.cv(space_params, 
                        dtrain=self.dtrain,
                        num_boost_round=500, 
                        nfold=k, 
                        stratified=True,  
                        early_stopping_rounds=30,
                        metrics = ['logloss'])
        
        best_score = results['test-logloss-mean'].min()
        return {'loss':best_score, 'status': STATUS_OK}
    def xgb_Kfold(self, k=5, max_evals=500):
        trials = Trials()
        best_hyperparams = fmin(fn=lambda params: self.xgb_obj(params, k=k), space=self.space, algo=tpe.suggest, max_evals=max_evals, trials=trials, early_stop_fn=no_progress_loss(30))
        best_params = space_eval(self.space, best_hyperparams)
        return best_params