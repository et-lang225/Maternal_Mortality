from sklearn.linear_model import LogisticRegressionCV
import numpy as np

class LR_class:
    def __init__(self):
        pass
    def LogRes(self, X, y):
        LR_mod = LogisticRegressionCV(Cs=10, l1_ratios=np.arange(0.05,0.95,0.05), cv=5, scoring='neg_log_loss', n_jobs=4, random_state=42)
        LR_fit = LR_mod.fit(X, y)
        return LR_fit