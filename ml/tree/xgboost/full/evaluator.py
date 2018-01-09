from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

class Evaluator(object):

    # Evaluator for special evaluations

    def __init__(self, index_col, target_col, param_c=2):
        self.param_c = param_c
        self.index_col = index_col
        self.target_col = target_col
        self.wrap_params = {
            'c': 200
        }

    def obj(self, preds, df_train):
        pass # abstract function here

    def eval_preds(self, preds, labels):
        return mean_absolute_error(preds, labels)

    def loss(self, labels, df_train):
        preds = df_train.get_label()
        return 'mae', self.eval_preds(self.unwrap(preds), self.unwrap(labels))       

    def wrap(self, labels):
        labels = np.log(labels + self.wrap_params['c'])
        return labels

    def unwrap(self, preds):
        preds = np.exp(preds) - self.wrap_params['c']
        return preds

class FairEvaluator(Evaluator):

    def obj(self, preds, df_train):
        labels = df_train.get_label()
        diff = preds - labels
        deno = abs(diff) + self.param_c
        grad = self.param_c * diff / deno
        hess = self.param_c ** 2 / deno ** 2
        return grad, hess
        
class CachyEvaluator(Evaluator):

    def obj(self, preds, df_train):
        labels = df_train.get_label()
        diff = preds - labels
        grad = diff / (diff ** 2 / self.param_c ** 2 + 1)
        hess = -self.param_c ** 2 * (x ** 2 - self.param_c ** 2) / (x**2 + self.param_c ** 2) ** 2
        return grad, hess

class LogcoshEvaluator(Evaluator):

    def obj(self, preds, df_train):
        labels = df_train.get_label()
        grad = np.tanh(preds - labels)
        hess = 1 - grad * grad
        return grad, hess
        
