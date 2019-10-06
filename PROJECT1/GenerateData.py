import numpy as np
from imageio import imread
import pandas as pd
import pathlib
from Config import cfg
import pandas as pd

def generate_terrain_data():
    data = imread(p.joinpath(cfg.RAW_TERRAIN_DATA_FILE))
    x1 = np.arange(data.shape[0])
    x2 = np.arange(data.shape[1])

    x1, x2 = np.meshgrid(x1,x2)

    x1=np.ravel(x1).reshape(-1,1)
    x2=np.ravel(x2).reshape(-1,1)
    X = np.concatenate((x1, x2), axis = 1)
    y = np.ravel(data)

    data = pd.DataFrame(np.concatenate((X, y.reshape(-1,1)), axis = 1), columns = ['x1', 'x2', 'y'])
    data.to_csv(cfg.TERRAIN_DATA)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def generate_franke_data(add_noise = False):
    x1 = np.sort(np.random.uniform(0, 1, cfg.DATA_SIZE))
    x2 = np.sort(np.random.uniform(0, 1, cfg.DATA_SIZE))

    x1, x2 = np.meshgrid(x1, x2)
    y = FrankeFunction(x1, x2)

    x1 = np.ravel(x1)
    x2 = np.ravel(x2)
    n = int(len(x1))
    y = np.ravel(y)

    filename = cfg.FRANKE_DATA

    if add_noise:
        y = y + np.random.randn(len(y))
        filename = cfg.FRANKE_DATA_NOISY
        
    X = np.concatenate([x1.reshape(-1,1), x2.reshape(-1,1), y.reshape(-1,1)], axis = 1)
    data = pd.DataFrame(X, columns = ['x1', 'x2', 'y'])
    data.to_csv(filename)


if __name__ == '__main__':

    p = cfg.DATA_BASE_PATH
    p.mkdir(parents=True, exist_ok=True)

    generate_terrain_data()
    generate_franke_data(add_noise = True)
    generate_franke_data(add_noise = False)


