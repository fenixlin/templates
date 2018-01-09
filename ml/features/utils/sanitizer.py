import pandas as pd
import numpy as np
import sys
import time
import collections

def run(filename):
    df = pd.read_csv(filename)
    nrow = df.shape[0]
    ncol = df.shape[1]
    saned_set = set(df.columns)
    print('[info] %s: Data loaded. Size:' % ('['+time.strftime("%H:%M:%S")+']'), nrow, 'x', ncol)
    
    # find pure null columns
    counter = df.isnull().sum()
    result = counter[counter==nrow].index
    print('[info] %s: Pure null value columns:' % ('['+time.strftime("%H:%M:%S")+']'), result.values)
    print('[info] %s: Number of pure null columns:' % ('['+time.strftime("%H:%M:%S")+']'), len(result))
    saned_set = saned_set.difference(result.values)

    # find strictly constant columns (includes null values)
    counter = df.apply(lambda s: s.nunique(dropna=False))
    result = counter[counter==1].index
    print('[info] %s: Strictly onstant columns:' % ('['+time.strftime("%H:%M:%S")+']'), result.values)
    print('[info] %s: Number of strictly constant columns:' % ('['+time.strftime("%H:%M:%S")+']'), len(result))

    # find loosely constant columns (includes null values)
    counter = df.apply(lambda s: s.nunique(dropna=True))
    result = counter[counter==1].index
    print('[info] %s: Loosely onstant columns:' % ('['+time.strftime("%H:%M:%S")+']'), result.values)
    print('[info] %s: Number of loosely constant columns:' % ('['+time.strftime("%H:%M:%S")+']'), len(result))
    saned_set = saned_set.difference(result.values)

    # find identical column pairs

    # find column pairs with projective relations

    # show columns with very few values

    # show columns with very few distinct values

    if len(saned_set)<ncol:
        print('[info] %s: Final recommendation of columns:' % ('['+time.strftime("%H:%M:%S")+']'))
        print(list(saned_set))

def run_large(filename):
    uni_val = collections.defaultdict(set)
    chunk_size = 10000
    chunk_id = 0
    null_count = None
    for df in pd.read_csv(filename, chunksize = chunk_size, low_memory=False):
        if chunk_id % 10 == 0:
            print('[info] %s: Processing chunk %d ...' % ('['+time.strftime("%H:%M:%S")+']', chunk_id))
        if chunk_id == 0:
            null_count = df.isnull().sum()
        else:
            null_count += df.isnull().sum()
        chunk_id += 1
        for col in df:
            uni_val[col] = uni_val[col].union(df[col][~df[col].isnull()].unique())
            

    cols = list(uni_val.keys())
    ncol = len(cols)
    saned_set = set(cols)

    result = list()
    for col in cols:
        if len(uni_val[col])==0:
            result.append(col)
    print('[info] %s: Pure null value columns:' % ('['+time.strftime("%H:%M:%S")+']'), result)
    print('[info] %s: Number of pure null columns:' % ('['+time.strftime("%H:%M:%S")+']'), len(result))
    saned_set = saned_set.difference(result)

    # find strictly constant columns (includes null values)
    result = list()
    for col in cols:
        if len(uni_val[col])==1 and null_count[col]==0:
            result.append(col)
    print('[info] %s: Strictly onstant columns:' % ('['+time.strftime("%H:%M:%S")+']'), result)
    print('[info] %s: Number of strictly constant columns:' % ('['+time.strftime("%H:%M:%S")+']'), len(result))

    # find loosely constant columns (includes null values)
    result = list()
    for col in cols:
        if len(uni_val[col])==1 and null_count[col]>0:
            result.append(col)
    print('[info] %s: Loosely onstant columns:' % ('['+time.strftime("%H:%M:%S")+']'), result)
    print('[info] %s: Number of loosely constant columns:' % ('['+time.strftime("%H:%M:%S")+']'), len(result))
    saned_set = saned_set.difference(result)

    # find identical column pairs

    # find column pairs with projective relations

    # show columns with very few values

    # show columns with very few distinct values

    if len(saned_set)<ncol:
        print('[info] %s: Final recommendation of columns:' % ('['+time.strftime("%H:%M:%S")+']'))
        print(list(saned_set))

if __name__ == '__main__':
    print('[info] %s: Please print output to files, as many columns will be recommended.' % ('['+time.strftime("%H:%M:%S")+']'))
    if len(sys.argv)>3 or len(sys.argv)<2:
        print('Example: python3 sanitizer.py ./input/train.csv -normal')
        sys.exit(0)
    print('[info] %s: Start sanitizing...' % ('['+time.strftime("%H:%M:%S")+']'))
    if len(sys.argv)==2 or sys.argv[2].strip('-')=='normal':
        run(sys.argv[1])
    elif sys.argv[2].strip('-')=='large':
        run_large(sys.argv[1])
    else:
        print('[ERROR] %s: Wrong option:' % ('['+time.strftime("%H:%M:%S")+']'))

