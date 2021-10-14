import pandas as pandas
import numpy as np
import xgboost as xgb


def simulate_data(n_obs, n_var):
    # create simulated data, with n_obs and n_var
    # returns X and y. y ~ X * beta, with random beta
    X = np.random.rand(n_obs, n_var)
    beta = np.random.rand(n_var, 1)
    y = np.matmul(X, beta)
    return y, X


def main(n_obs, n_var):
    y, X = simulate_data(n_obs, n_var)


if __name__ == 'main':
    n_obs = 10000
    n_var = 50
    main()
