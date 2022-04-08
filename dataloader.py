import pandas as pd
import numpy as np


def test_split(arr,lookback):
    n = arr.shape[0]
    r = n%lookback
    res_s = n-r
    return np.squeeze(np.split(arr[:res_s],res_s//lookback)),np.squeeze(arr[lookback:res_s:lookback])


def test_train_split(X,ratio = 0.8,lookback = 5):
    train = []
    test_X = []
    test_y = []
    for df in X:
        nparr = df.iloc[:,1].values[:,None]
        n =nparr.shape[0]
        train_s = int(ratio *n)
        test_s = n - train_s
        train.append(np.squeeze(nparr[:train_s]).astype(int))
        t_X,t_y = test_split(nparr[train_s:],lookback)
        test_X.append(t_X.astype(int))
        test_y.append(t_y.astype(int))
    return train,test_X,test_y
        