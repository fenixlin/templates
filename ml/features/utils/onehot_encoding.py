import pandas as pd
import sys

if __name__ == '__main__':
    if (len(sys.argv)<3):
        print('Usage: python3 onehot_encoding.py ../input/train.csv ../input/test.csv')
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
        
    result = pd.get_dummies(df[categorical_feat])
    print('Shape of one-hot encoded dataframe')
    print(result.shape)
    result.iloc[:split_point, :].to_csv('../cache/train_onehot_encoding.csv', index=False)
    result.iloc[split_point:, :].to_csv('../cache/test_onehot_encoding.csv', index=False)
        
