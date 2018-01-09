import pandas as pd
import numpy as np
import time
from sklearn.cross_validation import KFold, StratifiedKFold, StratifiedShuffleSplit
from model_util import save_preds

class ModelHolder(object):
    # encapsulating models with the ability to do cross validation prediction / data splitting prediction
    
    def __init__(self, config, target_col, index_col=None):
        self.valid_ratio = config['validation_ratio']
        self.cv_folds = config['cv_folds']
        self.shuffle_data = config['shuffle_data']
        self.stratified_valid = config['stratified_valid']
        self.random_state = config['random_state']
        self.target_col = target_col
        self.index_col = index_col
        self.model = None
        self.feature_imp = None
        self.train_score = None
        self.valid_score = None
        self.prediction = None

    def get_feature_imp(self):
        return self.feature_imp

    def get_train_score(self):
        return self.train_score

    def get_valid_score(self):
        return self.valid_score

    def get_prediction(self):
        return self.prediction
    
    def fit_predict(self, model, ori_train, ori_test=None, stack_prefix=None):
        if self.shuffle_data:
            ori_train = ori_train.iloc[np.random.permutation(ori_train.shape[0])].reset_index(drop=True)

        if self.cv_folds > 0 and isinstance(self.cv_folds, int):
            # (straight / stratified) cross validation
            print('[info] %s: Trainning with %d-fold cross validation started' % ('['+time.strftime("%H:%M:%S")+']', self.cv_folds))
            train_nrow = ori_train.shape[0]
            if ori_test is not None:
                test_nrow = ori_test.shape[0]
                test_preds = np.zeros((test_nrow, self.cv_folds))
            valid_preds = np.zeros(train_nrow)
            train_scores = np.zeros(self.cv_folds)
            imps = pd.Series()
            if self.stratified_valid:
                y = ori_train[self.target_col].values
                splitter = StratifiedKFold(y, nfolds = self.cv_folds, shuffle=True, random_state = self.random_state)
            else:
                splitter = KFold(train_nrow, n_folds = self.cv_folds)
            
            fold_i = -1
            for train_index, valid_index in splitter:
                fold_i += 1
                print('[info] %s: Fold %d/%d trainning started' % ('['+time.strftime("%H:%M:%S")+']', fold_i+1, self.cv_folds))
                df_train, df_valid = ori_train.iloc[train_index], ori_train.iloc[valid_index]
                model.fit(df_train, df_valid)
                if ori_test is not None:
                    test_preds[:, fold_i] = model.predict(ori_test)
                train_scores[fold_i] = model.get_train_score()
                valid_preds[valid_index] = model.predict(df_valid)
                imps = imps.add(model.get_feature_imp(), fill_value=0)
            # use median for MAE evaluation
            #self.prediction = np.median(test_preds, axis=1)
            if ori_test is not None:
                self.prediction = np.mean(test_preds, axis=1)
            if stack_prefix is not None:
                save_preds(stack_prefix+'_predict.csv', self.prediction, self.target_col)
                save_preds(stack_prefix+'_oob.csv', valid_preds, self.target_col)
            self.train_score = train_scores.mean()
            self.valid_score = model.get_evaluator().eval_preds(valid_preds, ori_train[self.target_col].values)
            imps.sort_values(ascending=False, inplace=True)
            self.feature_imp = imps
        else:
            if self.valid_ratio > 0 and self.valid_ratio < 1:
                print('[info] %s: Training with %f validation ratio started' % ('['+time.strftime("%H:%M:%S")+']', self.valid_ratio))
                # validation with a certain ratio
                train_nrow = ori_train.shape[0]
                df_train = df_valid = None
                if self.stratified_valid:
                    y = ori_train[self.target_col].values
                    x = ori_train.drop(self.target_col, axis=1).values
                    splitter = StratifiedShuffleSplit(y, 1, test_size=self.valid_ratio, random_state=self.random_state)
                    for train_index, valid_index in spliter:
                        df_train, df_valid = ori_train.iloc[train_index], ori_train.iloc[valid_index]
                else:
                    break_point = round(train_nrow * (1-self.valid_ratio))
                    df_train = ori_train.iloc[:break_point, :]
                    df_valid = ori_train.iloc[break_point:, :]
                
                model.fit(df_train, df_valid)
            else:
                # pure trainning
                print('[info] %s: Pure trainning started' % ('['+time.strftime("%H:%M:%S")+']'))
                model.fit(ori_train, None)

            self.train_score = model.get_train_score()
            self.valid_score = model.get_valid_score()
            self.feature_imp = model.get_feature_imp()
            if ori_test is not None:
                self.prediction = model.predict(ori_test)

