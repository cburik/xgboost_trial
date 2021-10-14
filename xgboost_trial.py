import numpy as np
import xgboost as xgb
import time


def simulate_data(n_obs, n_var):
    # create simulated data, with n_obs and n_var
    # returns X and y. y ~ X * beta, with random beta
    X = np.random.rand(n_obs, n_var)
    beta = np.random.rand(n_var, 1)
    y = np.matmul(X, beta)
    return y, X


def train_cpu(dmat):
    # train xgboost on cpu
    params = {'silent': 1}
    params['tree_method'] = 'hist'
    params['objective'] = 'reg:squarederror'
    start = time.time()
    xgb.train(params, dmat)
    end = time.time()
    return end - start


def train_gpu(dmat):
    # train xgboost on gpu
    params = {'silent': 1}
    params['tree_method'] = 'gpu_hist'
    params['objective'] = 'reg:squarederror'
    start = time.time()
    xgb.train(params, dmat)
    end = time.time()
    return end - start


def main(n_obs=10000, n_var=50):
    y, X = simulate_data(n_obs, n_var)
    dmat = xgb.DMatrix(X, label=y)
    time_cpu = train_cpu(dmat)
    print(time_cpu)
    time_gpu = train_gpu(dmat)
    print(time_gpu)


if __name__ == '__main__':
    n_obs = 100000
    n_var = 500
    main(n_obs, n_var)
