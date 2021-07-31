# import tensorflow as tf
import numpy as np


def generate_boundary_data(noise_level=0.05, N_u=20, X_mean=1, X_std=1, T_mean=1, T_std=1):
    XT_u1 = np.random.uniform(low=-1, high=1, size=(N_u, 1))
    Y_u1 = np.random.normal(loc=0, size=(N_u, 1)) * noise_level * np.exp(-3*np.abs(XT_u1)+1)
    # Y_u1 = np.sin(np.pi * (XT_u1 + Y_u1)) + Y_u1
    Y_u1 = np.sin(np.pi * XT_u1 ) + Y_u1
    XT_u1 = (XT_u1 - X_mean) / X_std
    T_u1 = (np.zeros(shape=(N_u, 1)) * 1.0 - T_mean) / T_std
    XT_u1 = np.hstack((XT_u1, T_u1))
    XTY_u1 = np.hstack((XT_u1, Y_u1))

    XT_u2 = np.random.uniform(low=0, high=1, size=(N_u//2, 1))
    # Y_u2 = np.random.normal(loc=0, size=(N_u//2, 1)) * noise_level * np.sqrt(np.abs(XT_u2))
    Y_u2 = np.zeros(shape=(N_u//2, 1))
    XT_u2 = (XT_u2 - T_mean) / T_std
    X_u2 = (np.ones(shape=(N_u//2, 1)) * (-1.0) - X_mean) / X_std
    XT_u2 = np.hstack((X_u2, XT_u2))
    XTY_u2 = np.hstack((XT_u2, Y_u2))

    XT_u3 = np.random.uniform(low=0, high=1, size=(N_u//2, 1))
    # Y_u3 = np.random.normal(loc=0, size=(N_u//2, 1)) * noise_level * np.sqrt(np.abs(XT_u3))
    Y_u3 = np.zeros(shape=(N_u//2, 1))
    XT_u3 = (XT_u3 - T_mean) / T_std
    X_u3 = (np.ones(shape=(N_u//2, 1)) * 1.0 - X_mean) / X_std
    XT_u3 = np.hstack((X_u3, XT_u3))
    XTY_u3 = np.hstack((XT_u3, Y_u3))

    XTY_u = np.concatenate((XTY_u1, XTY_u2, XTY_u3), axis=0)

    return np.float32(XTY_u)


def generate_interior_data(N_test=1000, X_mean=0, X_std=1, T_mean=0, T_std=1):
    X_test = (np.linspace(-1, 1, N_test) - X_mean) / X_std
    T_test = (np.linspace(1, 0, N_test) - T_mean) / T_std

    X_test = np.broadcast_to(X_test[None, :, None], (N_test, N_test, 1))
    T_test = np.broadcast_to(T_test[:, None, None], (N_test, N_test, 1))
    XT_test = np.concatenate((X_test, T_test), axis=-1)
    XT_test = np.reshape(XT_test, newshape=(-1, 2))

    return np.float32(XT_test)


def get_data(noise_level=0.05, N_r=200, N_u=20):
    # noise_level = 0.05

    X_r = np.random.uniform(low=-1, high=1, size=(N_r, 1))
    T_r = np.random.uniform(low=0, high=1, size=(N_r, 1))


    X_mean = np.mean(X_r)
    X_std = np.std(X_r)

    T_mean = np.mean(T_r)
    T_std = np.std(T_r)


    X_r = (X_r - X_mean) / X_std
    T_r = (T_r - T_mean) / T_std

    XT_r = np.hstack((X_r, T_r))
    XT_r = np.float32(XT_r)

    XTY_u = generate_boundary_data(noise_level, N_u, X_mean, X_std, T_mean, T_std)

    N_test = 100
    XT_test = generate_interior_data(N_test, X_mean, X_std, T_mean, T_std)

    return XTY_u, XT_r, XT_test, np.float32(X_mean), np.float32(X_std), np.float32(T_mean), np.float32(T_std)




