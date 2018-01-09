import pandas as pd
import sys
import numpy as np

if __name__ == '__main__':
    if (len(sys.argv)<2):
        print('Usage: python3 cate_distribution.py ../input/train.csv')
        exit(0)
    filename = sys.argv[1]
    df = pd.read_csv(filename)

    df['loss'] = np.log(df['loss'])
    categorical_feat = [col for col in df.columns if col.startswith('cat') and df[col].nunique()>2]
    #categorical_feat = [col for col in df.columns if df[col].dtype == 'O' and df[col].nunique()>2]
    for cate in categorical_feat:
        print(df.groupby(cate)['loss'].agg(['mean','max','min','nunique']))
        print('-------------------------------------')
        
