from Model import RidgeRegression, CrossVal, OLS
from Plots import bias_variance_plot, plot_3d_results
from DataProcessing import add_polynomials, load_data, generate_data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from Config import cfg

from utils import model_summary, test_different_polys_and_lambdas

DATA_SIZE = cfg.DATA_SIZE
POLY = cfg.POLY



if __name__ == "__main__":

    X, y = load_data()

    # Running trials for different polynomials
    polys = [i for i in range(1,25)]
    lambdas = [10**i for i in range(-3,8)]

    results = test_different_polys_and_lambdas(X = X, y = y, model_class = RidgeRegression, polys = polys, lambdas = lambdas)
    results.to_csv(cfg.RESULT_FOLDER.joinpath('ridge_polynomial_results.csv'))
    
    plot_3d_results(X = polys, Y = lambdas, results = results, save_as = "test_Ridge_lambda_poly_plot.png", metric = "test_mse")
    plot_3d_results(X = polys, Y = lambdas, results = results, save_as = "train_Ridge_lambda_poly_plot.png", metric = "train_mse")

    # Fitting best model getting results for that model 
    best_lambda = results[results.test_mse == results.test_mse.min()].lamb.tolist()[0]
    best_poly = results[results.test_mse == results.test_mse.min()].polynomials.tolist()[0]
    model = RidgeRegression(best_lambda)
    model_summary(model, add_polynomials(X, best_poly), y)

