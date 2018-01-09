import pandas as pd
import sys
import numpy as np

if __name__ == '__main__':
    if (len(sys.argv)<3):
        print('Usage: python3 cate_distribution.py ../input/train.csv ../input/test.csv')
        exit(0)
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    categorical_feat = [col for col in df_train.columns if col.startswith('cat')]
    #categorical_feat = [col for col in df_train.columns if df_train[col].dtype == 'O']
    for cate in categorical_feat:
        train_set = set(df_train[cate].unique())
        test_set = set(df_test[cate].unique())
        print('Feature', cate, ':')
        print('Only in train:', train_set.difference(test_set))
        print('Only in test:', test_set.difference(train_set))
        print('--------------')
        
