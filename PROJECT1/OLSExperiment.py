from Model import OLS, CrossVal
from Plots import bias_variance_plot
from DataProcessing import add_polynomials, load_data, generate_data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from Config import cfg
from utils import model_summary
import pathlib

DATA_SIZE = cfg.DATA_SIZE
POLY = cfg.POLY

def test_diffent_polynomials(X, y, polys = 10):
    polys = [i for i in range(1,polys)]
    results = []
    X_ = X
    for p in polys:
        X = X_
        X = add_polynomials(X, p)

        model = OLS()

        cross_validator = CrossVal(X, y, 10, model)
        cross_validator.fit()

        test_mse = np.mean(cross_validator.test_mses)
        train_mse = np.mean(cross_validator.train_mses)

        results.append([p,train_mse,test_mse])
    
    return  pd.DataFrame(results, columns = ['polynomials', 'train_mse', 'test_mse'])



if __name__ == "__main__":
    print("\n###   OLS  ### \n")

    X, y = load_data()
    
    model = OLS()
    # Running trials for different polynomials
    results = test_diffent_polynomials(X, y)
    results.to_csv(cfg.RESULT_FOLDER.joinpath('OLS_polynomial_results.csv'))

    best_poly = results[results.test_mse == results.test_mse.min()].polynomials.tolist()[0]

    model_summary(model, add_polynomials(X, best_poly), y)
    bias_variance_plot(results.polynomials, results.train_mse, results.test_mse, save = True)