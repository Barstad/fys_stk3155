import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from GenerateData import FrankeFunction
from Config import cfg
import pathlib
from sklearn.preprocessing import StandardScaler

PATH = cfg.ACTIVE_DATA

def add_polynomials(X, num_poly):
    poly = PolynomialFeatures(num_poly)
    X = poly.fit_transform(X)
    return X

def load_data(path = PATH):
    df = pd.read_csv(path)
    print(f"Loaded data from {path}")
    if df.shape[0] > 10000:
        df = df.sample(10000)
    # df = df.sample(n = cfg.USE_DATA_SIZE)
    X = df[['x1', 'x2']].values
    y = df.y.values.flatten()
    return X, y


# def FrankeFunction(x,y):
#     term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
#     term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
#     term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
#     term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
#     return term1 + term2 + term3 + term4

def generate_data(return_original = True):
    x1 = np.sort(np.random.uniform(0, 1, cfg.DATA_SIZE))
    x2 = np.sort(np.random.uniform(0, 1, cfg.DATA_SIZE))

    x1, x2 = np.meshgrid(x1, x2)
    y = FrankeFunction(x1, x2)

    x1 = np.ravel(x1)
    x2 = np.ravel(x2)
    n = int(len(x1))
    y = np.ravel(y)

    X_orig = np.concatenate([x1.reshape(-1,1), x2.reshape(-1,1)], axis = 1)
    X = add_polynomials(X_orig, 5)
    
    if return_original:
        return X, y, X_orig
    else:
        return X, y


def normalize_data(X_train, X_test, y_train, y_test):
    # Checking for constant column and removing it to keep it as ones
    condition = np.all(X_train[:,0] == 1)
    if condition:
        X_train = X_train[:,1:]
        X_test = X_test[:,1:]

    train_data = np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1)
    test_data = np.concatenate((X_test, y_test.reshape(-1,1)), axis = 1)

    # Scale data to 0 mean and 1 standard deviation
    scaler = StandardScaler(with_mean=True).fit(train_data)

    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    y_train = train_data[:,-1]
    y_test = test_data[:,-1]

    X_train = train_data[:,:-1]
    X_test = test_data[:,:-1]

    assert X_train.shape != train_data.shape

    # Adding back constant column if it was present
    if condition:
        X_train = np.concatenate([np.ones((len(X_train), 1)), X_train], axis = 1)
        X_test = np.concatenate([np.ones((len(X_test), 1)), X_test], axis = 1)

    return X_train, X_test, y_train, y_test


