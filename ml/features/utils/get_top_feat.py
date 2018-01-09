import pandas as pd

def print_list(filename, threshold):
    imp = pd.read_csv(filename, header=None)
    imp = imp[imp[2]>threshold]
    names = imp[1]
    print(list(names.values))


if __name__ == '__main__':
    #print_list('../output/feature_importance.csv', 100)
    print_list('../output/datediff_importance.csv', 35)

