from Model import RidgeRegression, CrossVal, OLS, LassoRegression
from sklearn.linear_model import Lasso

from Plots import bias_variance_plot, plot_3d_results
from DataProcessing import FrankeFunction, add_polynomials, generate_data, load_data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from Config import cfg

from utils import model_summary, test_different_polys_and_lambdas

POLY = cfg.POLY


if __name__ == "__main__":
    # X, y, X_orig = generate_data(return_original=True)

    # y = y + np.random.randn(len(X))

    X, y = load_data()

    # Running trials for different polynomials
    polys = [i for i in range(1,10)]
    lambdas = [10**i for i in range(-4,1)]

    results = test_different_polys_and_lambdas(X = X, y = y, model_class = LassoRegression, polys = polys, lambdas = lambdas)
    results.to_csv(cfg.RESULT_FOLDER.joinpath('lasso_polymial_results.csv'))
    plot_3d_results(X = polys, Y = lambdas, results = results, save_as = "Lasso_lambda_poly_plot.png")

    # Locating best model for 5 polonimals
    # min_error_for_5_poly = results[results.polynomials == 5].test_mse.min()
    # best_lambda_5_poly = results[(results.test_mse == min_error_for_5_poly) & (results.polynomials == POLY)].lamb.tolist()[0]
    
    best_lambda = results[results.test_mse == results.test_mse.min()].lamb.tolist()[0]
    best_poly = results[results.test_mse == results.test_mse.min()].polynomials.tolist()[0]
    model = RidgeRegression(best_lambda)
    model_summary(model, add_polynomials(X, best_poly), y)

