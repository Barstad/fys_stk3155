from Model import OLS, CrossVal
from DataProcessing import add_polynomials
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from Config import cfg
from DataProcessing import normalize_data
import pathlib

def model_summary(model, X, y, normalize = True):
    # regular train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    if normalize:
        X_train, X_test, y_train, y_test = normalize_data(X_train, X_test, y_train, y_test)
    
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = OLS.mse(train_pred, y_train)
    test_mse = OLS.mse(test_pred, y_test)
    train_r2 = OLS.R_squared(train_pred, y_train)
    test_r2 = OLS.R_squared(test_pred, y_test)

    print(f"Train mse : {train_mse}")
    print(f"Test mse : {test_mse}")
    print(f"Train R^2 : {train_r2}")
    print(f"Test R^2 : {test_r2}")

    results = pd.DataFrame(np.array([train_mse, test_mse, train_r2, test_r2]).reshape(1,4), columns = ['train mse', 'test mse', 'train R^2', 'test R^2'], index = ["Model Results"])
    results.to_csv(cfg.RESULT_FOLDER.joinpath(f"{model.name}_results.csv"))
    
    conf_int = model.confidence_interval_beta(X_train, y_train, 0.05)
    conf_int.to_csv(cfg.RESULT_FOLDER.joinpath(f"{model.name}_conf_int.csv"))
    print(f"95% confidence intervals for parameters: \n{conf_int}")


def test_different_polys_and_lambdas(X, y, model_class, polys, lambdas):
    
    results = []

    for p in polys:
        for k in lambdas:
            X_ = add_polynomials(X, p)

            model = model_class(k)

            cross_validator = CrossVal(X_, y, 10, model)
            cross_validator.fit()

            test_mse = np.mean(cross_validator.test_mses)
            train_mse = np.mean(cross_validator.train_mses)

            results.append([p,k,train_mse,test_mse])


    results = pd.DataFrame(results, columns = ['polynomials', 'lamb', 'train_mse', 'test_mse'])
    return results