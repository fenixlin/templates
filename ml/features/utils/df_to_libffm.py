import pandas as pd
import numpy as np
import time
import sys

def to_libffm_file(df, path, label=None):
    print('[info] %s: Writing to %s ...' % ('['+time.strftime("%H:%M:%S")+']', path))
    ffm = open(path,'w')
    for i in range(df.shape[0]):
        row = df.iloc[i,:]
        if label is not None:
            ffm.write(str(label.iloc[i]))
        else:
            ffm.write('0')
        row = row.reset_index(drop=True)
        for j in range(len(row)):
            ffm.write(' %d:%d:%d' % (int(row.index[j]), row.iloc[j], 1))
        ffm.write('\n')
        if i % 20000 == 0:
            print('[info] %s: %d rows proceeded.' % ('['+time.strftime("%H:%M:%S")+']', i))
            
    ffm.close()

if __name__ == '__main__':
    if (len(sys.argv)<3):
        print('Usage: python3 df_to_libffm.py ../input/train.csv ../input/test.csv')
        exit(0)
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    df = pd.concat([df_train, df_test]).reset_index(drop=True)
    split_point = df_train.shape[0]
    categorical_feat = [col for col in df.columns if df[col].dtype == 'O']
    print('Categorical features:')
    print(categorical_feat)

    # encode categorical features & transform labels
    for cate in categorical_feat:
        df[cate] = df[cate].factorize(sort=True)[0]
    df['loss'] = np.log(df['loss']+200)

    # write to ffm format file
    df_train = df.iloc[:split_point, :]
    df_test = df.iloc[split_point:, :]
    to_libffm_file(df_train[categorical_feat], '../cache/train.ffm', df_train['loss'])
    to_libffm_file(df_test[categorical_feat], '../cache/test.ffm')
