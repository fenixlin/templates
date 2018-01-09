import json
import time
import pandas as pd
import hyperopt as hp
import model
from model_holder import ModelHolder
from model_util import save_preds

STACK_PREFIX = './stack/level1_stack' #None
IMP_SAVEFILE = './stack/level1_stack.imp' #'./output/feature_importance.csv' 

def run_model(df_train, df_test, config, save_imp=True, params=None):
    # run model with trainning & predicting process
    # Warning: drop_list columns should be dropped already
    mdf = model.ModelFactory(target_col=config['target_col'], index_col=config['index_col'])
    mdl = mdf.build_model(config, params)
    mdh = ModelHolder(config, target_col=config['target_col'], index_col=config['index_col'])
    mdh.fit_predict(mdl, df_train, df_test, stack_prefix=STACK_PREFIX)
    imps = mdh.get_feature_imp()
    if (save_imp) and (imps is not None):
        imps.to_csv(IMP_SAVEFILE, header=None)
    return mdh

def produce_model(config, drop_list, maximize=False):
    param_space_dict = dict()
    param_space_dict['xgbr'] = {
        'booster' : 'gbtree',
        'eta' : hp.hp.uniform('eta', 0.005, 0.03),
        'max_depth': hp.hp.choice('max_depth', [9,10,11,12,13]),
        'subsample' : hp.hp.uniform('subsample', 0.7, 0.85),
        'colsample_bytree' : hp.hp.uniform('colsample_bytree', 0.7, 0.85),
        'min_child_weight' : hp.hp.uniform('min_child_weight', 1, 150),
        'silent': 1,
        'nthread': 9,
    }
    df_train = pd.read_csv(config['train_file'])
    df_test = pd.read_csv(config['test_file'])
    test_indices = df_test[config['index_col']].values
    df_train = df_train.drop(drop_list, axis=1)
    df_test = df_test.drop(drop_list, axis=1)
    best_score = 999999
    if maximize:
        best_score = -999999
    best_param = None
    trial_counter = 0

    def one_trial(params):
        nonlocal config
        nonlocal df_train
        nonlocal df_test
        nonlocal test_indices
        nonlocal trial_counter
        nonlocal best_score
        nonlocal best_param
        nonlocal drop_list
        nonlocal maximize
        
        trial_counter += 1
        print('[info] %s: trial %d' % ('['+time.strftime("%H:%M:%S")+']', trial_counter))
        mdh = run_model(df_train, df_test, config, save_imp=False, params=params)
        score = mdh.get_valid_score()
        if (not maximize and score < best_score) or (maximize and score > best_score):
            best_score = score
            best_param = params
        model_name = config['model']+'_hpo_'+str(trial_counter)
        test_preds = mdh.get_prediction()
        save_preds('./output/'+model_name+'.csv', test_preds, config['target_col'], test_indices, config['index_col'])
        print('[info] params : %s' % str(params))
        print('[info] new score %s, best score %s' % (str(score), str(best_score)))
        print('----------')
        return score

    param_space = param_space_dict[config['model']]
    trials = hp.Trials()
    best_result = hp.fmin(one_trial, param_space, algo=hp.tpe.suggest, trials=trials, max_evals=config['hyperopt_num'])
    print("Best score: "+str(best_score))
    print("Best param: "+str(best_param))
    print("Best result: "+str(best_result))
    _, trial_result = min(enumerate(trials.results), key=lambda k: k[1]['loss'])
    print("Best trial results:")
    print(trial_result)

def run(config, drop_list):
    print('[info] %s: Model Start Running' % ('['+time.strftime("%H:%M:%S")+']'))
    # read data file
    df_train = pd.read_csv(config['train_file'])
    df_test = pd.read_csv(config['test_file'])
    test_indices = df_test[config['index_col']].values
    df_train = df_train.drop(drop_list, axis=1)
    df_test = df_test.drop(drop_list, axis=1)

    print('[info] %s: Data Loaded. Train size %d x %d. Test size %d x %d' % ('['+time.strftime("%H:%M:%S")+']', df_train.shape[0], df_train.shape[1], df_test.shape[0], df_test.shape[1]))
    # construct and run model
    mdh = run_model(df_train, df_test, config, save_imp=True)

    print('[info] %s: Model trained' % ('['+time.strftime("%H:%M:%S")+']'))
    # print evaluation and save result
    print('[info] %s: Trainning score:' % ('['+time.strftime("%H:%M:%S")+']'), mdh.get_train_score())
    print('[info] %s: Validation score:' % ('['+time.strftime("%H:%M:%S")+']'), mdh.get_valid_score())
    test_preds = mdh.get_prediction()
    save_preds('./output/submission.csv', test_preds, config['target_col'], test_indices, config['index_col'])
    print('[info] %s: Test Data Predicted' % ('['+time.strftime("%H:%M:%S")+']'))

if __name__ == '__main__':
    config = json.load(open('./config.json', 'r'))
    drop_list = [config['index_col']]
    if ('hyperopt_num' in config) and (config['hyperopt_num']>0):
        produce_model(config, drop_list)
    else:
        run(config, drop_list)
