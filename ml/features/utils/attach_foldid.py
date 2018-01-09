import pandas as pd
import numpy as np
import time
import json
import sys
from sklearn.cross_validation import KFold, StratifiedKFold, StratifiedShuffleSplit

config = json.load(open('../cache/config.json', 'r'))
STRATIFIED = False
SHUFFLE = config['shuffle_data'] # 'True' is recommended
NFOLD = 10
FOLD_COL = 'fold_id'
TARGET_COL = 'loss'
RANDOM_STATE = config['random_state']

if __name__ == '__main__':
    if (len(sys.argv)<2):
        print('Usage: python3 attach_foldid.py ../input/train.csv')
        exit(0)
    filename = sys.argv[1]
    df = pd.read_csv(filename)

    print('[info] %s: %d-fold splitting started' % ('['+time.strftime("%H:%M:%S")+']', NFOLD))
    nrow = df.shape[0]
    df[FOLD_COL] = np.zeros(nrow).astype(int)
    if STRATIFIED:
        y = df[TARGET_COL].values
        splitter = StratifiedKFold(y, nfolds = NFOLD, shuffle = SHUFFLE, random_state = RANDOM_STATE)
    else:
        splitter = KFold(nrow, n_folds = NFOLD, shuffle = SHUFFLE, random_state = RANDOM_STATE)
    
    fold_i = 0
    for train_index, valid_index in splitter:
        fold_i += 1
        df.ix[valid_index, FOLD_COL] = fold_i

    df.to_csv(filename[:-4]+'_folded.csv', index=False)
    print('[info] %s: %d-fold splitting done' % ('['+time.strftime("%H:%M:%S")+']', NFOLD))
