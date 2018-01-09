import xgboost as xgb
import scipy.sparse as sp
import pandas as pd
import numpy as np
from evaluator import *
import xgbfir

class ModelFactory(object):
    def __init__(self, target_col, index_col=None):
        self.model_dict = {
            'xgbr': XgboostRegression(target_col, index_col)
        }

    def build_model(self, config, params=None):
        model = self.model_dict[config['model']]
        if params is not None:
            model.set_params(params)
        model.set_config(config)
        return model

class Model(object):
    def __init__(self, target_col, index_col):
        self.model = None
        self.config = None
        self.params = None
        self.train_score = None
        self.valid_score = None
        self.feature_imp = None
        self.evaluator = None
        self.target_col = target_col
        self.index_col = index_col

    def fit(self, df_train, df_valid=None):
        print('abstract method')

    def predict(self, df_test):
        print('abstract method')

    def set_config(self, config):
        self.config = config

    def set_target_col(self, target_col):
        self.target_col = target_col

    def set_index_col(self, index_col):
        self.index_col = index_col

    def set_params(self, params):
        self.params = params
        for k,v in self.params.items():
            self.model.set_params(**{k: v})

    def get_valid_score(self):
        return self.valid_score

    def get_train_score(self):
        return self.train_score

    def get_feature_imp(self):
        return self.feature_imp

    def get_evaluator(self):
        return self.evaluator

class XgboostRegression(Model):

    def __init__(self, target_col, index_col):
        Model.__init__(self, target_col, index_col)
        self.evaluator = FairEvaluator(self.index_col, self.target_col, param_c = 2)
        self.params = {
            'booster' : 'gbtree',
            'eta' : 0.03,
            'max_depth' : 12,
            'min_child_weight' : 100,
            'subsample' : 0.7,
            'colsample_bytree' : 0.7,
            'silent' : 1,
            'nthread' : 9
#            'base_score' : 7.657
#            'eval_metric' : 'mae',
#            'objective' : 'reg:linear',
        }
        self.num_rounds = 100000
        self.early_stopping = 50
        self.verbose_eval = 200
        self.fmap_file = './cache/feature.map'

    def set_params(self, params):
        self.params = params

    def _output_fmap(self, df_train):
        f = open(self.fmap_file, 'w')
        names = df_train.columns
        for i in range(len(names)):
            f.write('%d\t%s\ti\n' % (i, names[i]))
        f.close()

    def fit(self, df_train, df_valid=None):
        train_label = df_train[self.target_col].values
        if self.config['wrap_data']:
            train_label = self.evaluator.wrap(train_label)

        df_train = df_train.drop([self.target_col], axis=1)
        self._output_fmap(df_train)
        train_mat = xgb.DMatrix(df_train, label=train_label, silent=True)
        score_dict = dict()
        if df_valid is None:
            self.model = xgb.train(params=self.params, dtrain=train_mat, num_boost_round=self.num_rounds, evals=[(train_mat, 'train')], early_stopping_rounds=self.early_stopping, verbose_eval=self.verbose_eval, evals_result=score_dict, obj=self.evaluator.obj, feval=self.evaluator.loss)
        else:
            valid_label = df_valid[self.target_col].values
            if self.config['wrap_data']:
                valid_label = self.evaluator.wrap(valid_label)
            df_valid = df_valid.drop([self.target_col], axis=1)
            valid_mat = xgb.DMatrix(df_valid, label=valid_label, silent=True)
            self.model = xgb.train(params=self.params, dtrain=train_mat, num_boost_round=self.num_rounds, evals=[(train_mat, 'train'), (valid_mat, 'valid')], early_stopping_rounds=self.early_stopping, verbose_eval=self.verbose_eval, evals_result=score_dict, obj=self.evaluator.obj, feval=self.evaluator.loss)
            valid_score_list = score_dict['valid']['mae']
            self.valid_score = self.model.best_score
            
        train_score_list = score_dict['train']['mae']
        self.train_score = train_score_list[len(train_score_list)-1]
        imps = pd.DataFrame(list(self.model.get_fscore(fmap=self.fmap_file).items()))
        imps.set_index(0, drop=True, inplace=True)
        imps.sort_values(1, ascending=False, inplace=True)
        self.feature_imp = imps[1]

        #xgbfir.saveXgbFI(self.model, feature_names=df_train.columns, OutputXlsxFile='./cache/xgbfir_interaction.xlsx')

    def predict(self, df_test):
        # drop y in validation sets
        if self.target_col in df_test.columns:
            df_test = df_test.drop([self.target_col], axis=1)

        test_mat = xgb.DMatrix(df_test, silent=True)
        test_preds = self.model.predict(test_mat)
        if self.config['wrap_data']:
            test_preds = self.evaluator.unwrap(test_preds)
        return test_preds

