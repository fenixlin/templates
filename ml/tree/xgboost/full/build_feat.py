import json
import time
import pandas as pd
import numpy as np
import os

def ismade(df, feat_names, anyof=True):
    if anyof:
        if feat_names[0] in df.columns:
            return True
        else:
            print('[info] %s: %s not made!' % ('['+time.strftime("%H:%M:%S")+']', feat_names))
            return False
    else:
        check = True
        for it in feat_names:
            if it not in df.columns:
                check = False
                break
        if not check:
            print('[info] %s: %s not made!' % ('['+time.strftime("%H:%M:%S")+']', feat_names))
        return check

#def make_indv_feat(df, ext_files=None):
#    '''
#        get feature for either train or test
#    '''
#    feats = pd.DataFrame()
#    origin_feats = df[FEAT_LIST]
#    if 'Response' in origin_feats.columns:
#        origin_feats = origin_feats.drop('Response', axis=1)
#
#    # count nulls, and their positions
#    if not ismade(df, ['nonnulls','nonnull_min','nonnull_max','nonnull_mean','nonnull_std']):
#        feats = count_nonnull(origin_feats, feats)
#
#    new_cols = feats.columns.difference(df.columns)
#    if len(new_cols)>0:
#        df = df.merge(feats[new_cols], left_index=True, right_index=True)
#    return df

def make_combined_feat(df_train, df_test, ext_files=None):
    '''
        get feature from combined train and test dataframe
    '''
    df = pd.concat([df_train, df_test]).reset_index(drop=True)
    split_point = df_train.shape[0]

    ######
    # code for feature building
    ######
    # usage:
    #    if not ismade(df, ['nonnulls','nonnull_min','nonnull_max','nonnull_mean','nonnull_std']):
    #       make_feature()
    ######

    return df.iloc[:split_point, :], df.iloc[split_point:, :]

def run(config):
    print('[info] %s: Start Making Features.' % ('['+time.strftime("%H:%M:%S")+']'))

    # load train & test file, make features progressively if former file available
    print('[info] %s: Loading data...' % ('['+time.strftime("%H:%M:%S")+']'))
    index_col = config['index_col']
    if os.path.isfile(config['train_file']):
        df_train = pd.read_csv(config['train_file'])
    else:
        df_train = pd.read_csv(config['train_source'])
    if os.path.isfile(config['test_file']):
        df_test = pd.read_csv(config['test_file'])
    else:
        df_test = pd.read_csv(config['test_sourse'])
    print('[info] %s: Data Loaded' % ('['+time.strftime("%H:%M:%S")+']'))

    # get individual feature
#    df_train = make_indv_feat(df_train)
#    print('[info] %s: Individual feature for trainning set computed.' % ('['+time.strftime("%H:%M:%S")+']'))
#    df_test = make_indv_feat(df_test)
#    print('[info] %s: Individual feature for test set computed.' % ('['+time.strftime("%H:%M:%S")+']'))

    # get combined feature
    df_train, df_test = make_combined_feat(df_train, df_test)
    print('[info] %s: Combined feature for whole data set computed.' % ('['+time.strftime("%H:%M:%S")+']'))

    # save all features
    save_order = df_train.columns.drop(index_col).insert(0, index_col)
    df_train.to_csv(train_target, index=False, columns=save_order)
    save_order = df_test.columns.drop(index_col).insert(0, index_col)
    df_test.to_csv(test_target, index=False, columns=save_order)
    print('[info] %s: Finish Making Features' % ('['+time.strftime("%H:%M:%S")+']'))

if __name__ == '__main__':
    config = json.load(open('./config.json', 'r'))
    run(config)
