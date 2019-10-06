from Plots import show_regularization_effects
from Model import LassoRegression, RidgeRegression
from DataProcessing import load_data, add_polynomials, normalize_data
from sklearn.model_selection import train_test_split



if __name__ == '__main__':
    X, y = load_data()
    X = add_polynomials(X, 5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_train, X_test, y_train, y_test = normalize_data(X_train, X_test, y_train, y_test)

    show_regularization_effects(X_train, y_train, LassoRegression, lower = -4, upper = 1)
    show_regularization_effects(X_train, y_train, RidgeRegression, lower = -4, upper = 3)
