import tensorflow as tf
import numpy as np


def get_data(noise_level=0.05, N_r=200, N_u=20):
    # noise_level = 0.05

    X_r = np.linspace(-1., 1., N_r)[:,None]
    X_mean, X_std = np.mean(X_r), np.std(X_r)
    X_r = (X_r - X_mean) / X_std

    X_u = np.vstack((np.ones(shape=(N_u, 1)) * 1.0, np.ones(shape=(N_u, 1)) * (-1.0)))
    Y_u = np.random.normal(size=(N_u * 2, 1)) * noise_level
    X_u = (X_u - X_mean) / X_std
    XY_u = np.hstack((X_u, Y_u))

    N_test = 2000
    X_test = np.linspace(-1., 1., N_test)[:,None]
    X_test = (X_test - X_mean) / X_std

    XY_u = np.float32(XY_u)
    X_r = np.float32(X_r)
    X_test = np.float32(X_test)
    X_mean = np.float32(X_mean)
    X_std = np.float32(X_std)

    return XY_u, X_r, X_test, X_mean, X_std
