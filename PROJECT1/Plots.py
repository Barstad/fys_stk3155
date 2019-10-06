from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pathlib
from Config import cfg

PATH = cfg.RESULT_FOLDER

def bias_variance_plot(complexity, train_error, test_error, save = False):
    fig, ax = plt.subplots(figsize = (10,7))
    plt.plot(complexity, train_error, label = "Train error")
    plt.plot(complexity, test_error, label = "Test error")
    plt.legend()
    plt.ylabel("Prediction Error")
    plt.xlabel("Model Complexity")
    x_ticks = ['' for i in range(len(complexity))]
    x_ticks[0] = 'Low'
    x_ticks[-1] = 'High'
    plt.xticks(ticks = complexity, labels = x_ticks)
    yticks = plt.yticks()
    label_values = ['' for i in range(len(yticks[0]))]
    plt.yticks(ticks = yticks[0], labels = label_values)
    plt.plot()
    if save:
        fig.savefig(PATH.joinpath("BiasVaricancePlot.png"))


def plot_3d_results(X, Y, results, save_as, metric = 'test_mse'):
    def fun(x, y):
        row = results[(results.polynomials == x) & (results.lamb == y)]
        return row[metric].tolist()[0]

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(X, Y)
    zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Y = np.log10(Y)

    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, label = 'Test')

    ax.set_xlabel('Polynomials')
    ax.set_ylabel('Lambda (Log10)')
    ax.set_zlabel('Test Error')
    
    plt.show()
    fig.savefig(PATH.joinpath(save_as))


def show_regularization_effects(X, y, model_class, upper = 3, lower = -4):
    r = range(lower,upper)
    lambdas = [10**i for i in r]
    
    coefs_list = []
    for l in lambdas:
        model = model_class(l)
        model.fit(X, y)
        coefs = list(model.get_betas(X, y))
        coefs_list.append(coefs)

    coefs_list = np.array(coefs_list)
    fig, ax = plt.subplots(figsize = (15,5))
    for i in range(coefs_list.shape[1]):
        plt.plot([i for i in r], coefs_list[:,i])
    plt.title("Effect of lambda on the size of beta")
    plt.xlabel("Lambda (log10)")
    plt.ylabel("Beta")
    plt.show()
    fig.savefig(PATH.joinpath(f"lambda_effect_on_beta_for_{model.name}.png"))

