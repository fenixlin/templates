import pandas as pd

def save_preds(filename, preds, target_col, indices=None, index_col=None):
    if (indices is not None) and (index_col is not None):
        pd.DataFrame({index_col: indices, target_col: preds}).to_csv(filename, index=False)
    else:
        pd.DataFrame({target_col: preds}).to_csv(filename, index=False)

