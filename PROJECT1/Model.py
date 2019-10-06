import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler
from scipy.stats import t
from DataProcessing import normalize_data
from sklearn.linear_model import Lasso


class OLS:
    def __init__(self):
        self.fitted = False
        self.name = 'OLS'
        
    def get_betas(self, X, y):        
        beta = np.linalg.inv((X.T).dot(X)).dot((X.T).dot(y))
        return beta

    def fit(self, X, y):        
        self.betas = self.get_betas(X, y)
        self.fitted = True
        
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.predict(X)
    
    def predict(self, X):
        y_hat = X.dot(self.betas)
        return y_hat
    
    @staticmethod
    def mse(pred, actual):
        return np.mean(np.square(pred - actual))
    
    @staticmethod
    def R_squared(pred, actual):
        r_sq = 1 - OLS.mse(pred, actual)/ OLS.mse(actual.mean(), actual)
#         print("R^2 : {}".format(r_sq))
        return r_sq

    def cov_beta(self, X, y):
        return np.linalg.inv((X.T).dot(X)) * self.mse(self.fit_transform(X, y), y) * \
    len(X)/(len(X) - X.shape[1])

    def var_beta(self, X, y):
        return np.diag(self.cov_beta(X, y))

    def confidence_interval_beta(self, X, y, alpha):

        limit = alpha/2.0 * 100
        
        var = self.var_beta(X, y)
        beta = self.betas
        
        T = t.ppf(1-alpha/2.0, df = X.shape[0] - X.shape[1])
        
        upper_bound = beta + T * np.sqrt(var)
        lower_bound = beta - T * np.sqrt(var)
        
        coefficients = np.concatenate([np.expand_dims(lower_bound, 1), 
                               np.expand_dims(upper_bound, 1)],
                              axis = 1)

        coefs = pd.DataFrame(coefficients, columns = [f"{limit}", f"{100-limit}"])
        coefs.index = [f"beta_{i}" for i in range(X.shape[1])]
        return coefs
    

class RidgeRegression(OLS):
    def __init__(self, l2_penalty):
        self.l2_penalty = l2_penalty
        self.fitted = False
        self.name = 'Ridge'
        
    def get_betas(self, X, y):
        beta = np.linalg.inv((X.T).dot(X) + self.l2_penalty * np.eye(X.shape[1])).dot((X.T).dot(y))
        return beta

    def cov_beta(self, X, y):
        term1 = np.linalg.inv((X.T).dot(X) + self.l2_penalty * np.eye(X.shape[1]))
        term2 = (X.T).dot(X)
        term3 = self.mse(self.fit_transform(X, y), y) * len(X)/(len(X) - X.shape[1])
        
        return term3 * term1.dot(term2).dot(term1)


class LassoRegression():
    def __init__(self, l1_penalty):
        self.l1_penalty = l1_penalty
        self.name = 'Lasso'
        self.fitted = False
        self.model = Lasso(l1_penalty, fit_intercept=False)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X):
        return self.model.predict(X)

    def get_betas(self, X, y):
        return self.model.coef_

    def confidence_interval_beta(self, X, y, alpha = 0.05, trials = 1000):
        alpha = 1-alpha
        alpha = alpha * 100
        limit = (100-alpha)/2.0

        ind = np.arange(len(X))
        N = len(X)

        coefficients = []
        for i in range(trials):
            ind_ = np.random.choice(ind, size = N)
            X_ = X[ind_]
            y_ = y[ind_]
            self.model.fit(X_, y_)

            coef = self.model.coef_.tolist()
            coefficients.append(coef)


        coefs = pd.DataFrame(coefficients)
        coefs = coefs.apply(lambda x: np.percentile(x, [limit, alpha + limit]), axis = 0).T
        coefs.columns = [f"{limit}", f"{100-limit}"]
        coefs.index = [f"beta_{i}" for i in range(X.shape[1])]
        return coefs


# Class to run cross validation for given dataset, k-folds and model-class (e.g. OLS)
class CrossVal():  
    def __init__(self, X, y, k, model, normalize = True):
        self.X = X
        self.y = y
        self.k = k
        self.index = np.array([i for i in range(len(X))])
        self.model = model
        self.normalize = normalize
        self.k_fold_index_splitter()

    def k_fold_index_splitter(self):
        X = self.X
        k = self.k
        sample_size = int(len(X) / k)
        data_indecies = np.random.choice(a=len(X), size=(k - 1, sample_size), replace=False)
        kth_fold = np.array(list(idx for idx in list(range(0, len(X))) if idx not in data_indecies.flatten()))
        k_fold_indecies = list(data_indecies)
        k_fold_indecies.append(kth_fold)
        self.k_fold_indecies = k_fold_indecies
        
    def mse(self, pred, actual):
        return (np.mean(np.square(pred - actual)))

    def fit(self):
        model = self.model
        
        models = []
        train_mses = []
        test_mses = []
        train_r_sqs = []
        test_r_sqs = []        
        
        for kth in range(self.k):
            
            X_train = self.X[~np.isin(self.index, self.k_fold_indecies[kth])]
            y_train = self.y[~np.isin(self.index, self.k_fold_indecies[kth])]
            
            X_test = self.X[np.isin(self.index, self.k_fold_indecies[kth])]
            y_test = self.y[np.isin(self.index, self.k_fold_indecies[kth])]
            
            if self.normalize:
                X_train, X_test, y_train, y_test = normalize_data(X_train, X_test, y_train, y_test)

            
            model.fit(X_train, y_train)
            models.append(copy.deepcopy(model))
            test_pred = model.predict(X_test)
            train_pred = model.predict(X_train)

            test_mse_ = self.mse(test_pred, y_test)
            test_mses.append(test_mse_)
            train_mse_ = self.mse(train_pred, y_train)
            train_mses.append(train_mse_)
            
            test_r_sq = 1 - test_mse_ / self.mse(y_test.mean(), y_test)
            test_r_sqs.append(test_r_sq)
            train_r_sq = 1 - train_mse_ / self.mse(y_train.mean(), y_train)
            train_r_sqs.append(train_r_sq)
            
            
        self.models = models
        self.train_mses = train_mses
        self.test_mses = test_mses
        
        self.train_r_sqs = train_r_sqs
        self.test_r_sqs = test_r_sqs
        
        self.fitted = True