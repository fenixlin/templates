import pandas as pd
import sys

def print_df(df, step):
    for i in range(min(step, len(df.columns)), len(df.columns)+1, step):
        print(df.iloc[:, i-step:i].describe())
        print('-------------------------------------')
    if len(df.columns) % step >0:
        print(df.iloc[:, len(df.columns)//step*step : len(df.columns)].describe())

if __name__ == '__main__':
    if (len(sys.argv)<2):
        print('Usage: python3 basic_describe.py ../input/train.csv')
        exit(0)
    filename = sys.argv[1]
    df = pd.read_csv(filename)

    print('-------------------------------------')
    print(' PRINTING NUMERICAL FEATURES ')
    print('-------------------------------------')
    numerical_feat = [col for col in df.columns if df[col].dtype in ('int64', 'float64')]
    print_df(df[numerical_feat], step=9)

    print('-------------------------------------')
    print(' PRINTING CATEGORICAL FEATURES ')
    print('-------------------------------------')
    categorical_feat = [col for col in df.columns if df[col].dtype == 'O']
    print_df(df[categorical_feat], step=9)
        
