from sklearn import datasets
import pandas as pd
import numpy as np

def load_dataset(flowers: list, features: list):
    iris = datasets.load_iris()

    df_tn = pd.DataFrame(iris.target_names)
    df_tn = df_tn[df_tn.isin(flowers)]
    df_tn = df_tn.dropna()
    idx_tn = df_tn.index[:] # returns 0, 1

    df_y = pd.DataFrame(iris.target)
    df_y = df_y[df_y.isin(idx_tn.values)]
    df_y = df_y.dropna()
    # df_y = df_y.where(df_y==1, -1)
    idx_y = df_y.index[:] # returns 1..100

    df_fn = pd.DataFrame(iris.feature_names)
    df_fn = df_fn[~df_fn.isin(features)]
    df_fn = df_fn.dropna()
    idx_fn = df_fn.index[:] # returns 2, 3

    df_x = pd.DataFrame(iris.data)
    df_x = df_x.iloc[idx_y.values]
    df_x = df_x.drop(labels=idx_fn, axis=1)
  
    
    y = df_y.values
    X = df_x.values
    return X, y

def load_dataset_std(flowers: list, features: list):
    X, y = load_dataset(flowers, features)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std
    return X, y