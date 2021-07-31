import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def generate_boundary_data(noise_level=0.05, N_u=20, X_mean=0, X_std=1, Y_mean=0, Y_std=1):
    X_u1 = np.random.uniform(low=-1, high=1, size=(N_u,1))
    Y_u1 = np.ones(shape=(N_u,1)) * (-1.0)
    U1 = np.random.normal(loc=0, size=(N_u, 1)) * noise_level
    X_u1 = (X_u1 - X_mean) / X_std
    Y_u1 = (Y_u1 - Y_mean) / Y_std
    XY_u1 = np.concatenate((X_u1, Y_u1), axis=-1)
    XYU_u1 = np.concatenate((XY_u1, U1), axis=-1)

    X_u2 = np.random.uniform(low=-1, high=1, size=(N_u, 1))
    Y_u2 = np.ones(shape=(N_u,1)) * (1.0)
    U2 = np.random.normal(loc=0, size=(N_u, 1)) * noise_level
    X_u2 = (X_u2 - X_mean) / X_std
    Y_u2 = (Y_u2 - Y_mean) / Y_std
    XY_u2 = np.concatenate((X_u2, Y_u2), axis=-1)
    XYU_u2 = np.concatenate((XY_u2, U2), axis=-1)

    Y_u3 = np.random.uniform(low=-1, high=1, size=(N_u, 1))
    X_u3 = np.ones(shape=(N_u, 1)) * (-1.0)
    U3 = np.random.normal(loc=0, size=(N_u, 1)) * noise_level
    X_u3 = (X_u3 - X_mean) / X_std
    Y_u3 = (Y_u3 - Y_mean) / Y_std
    XY_u3 = np.concatenate((X_u3, Y_u3), axis=-1)
    XYU_u3 = np.concatenate((XY_u3, U3), axis=-1)

    Y_u4 = np.random.uniform(low=-1, high=1, size=(N_u, 1))
    X_u4 = np.ones(shape=(N_u, 1)) * (1.0)
    U4 = np.random.normal(loc=0, size=(N_u, 1)) * noise_level
    X_u4 = (X_u4 - X_mean) / X_std
    Y_u4 = (Y_u4 - Y_mean) / Y_std
    XY_u4 = np.concatenate((X_u4, Y_u4), axis=-1)
    XYU_u4 = np.concatenate((XY_u4, U4), axis=-1)

    XYU_u = np.concatenate((XYU_u1, XYU_u2, XYU_u3, XYU_u4), axis=0)

    return np.float32(XYU_u)


def generate_interior_data(N_test=1000, X_mean=0, X_std=1, Y_mean=0, Y_std=1):
    X_test = (np.linspace(-1, 1, N_test) - X_mean) / X_std
    Y_test = (np.linspace(-1, 1, N_test) - Y_mean) / Y_std

    X_test = np.broadcast_to(X_test[None, :, None], (N_test, N_test, 1))
    Y_test = np.broadcast_to(Y_test[:, None, None], (N_test, N_test, 1))
    XY_test = np.concatenate((X_test, Y_test), axis=-1)
    XY_test = np.reshape(XY_test, newshape=(-1, 2))

    return np.float32(XY_test)


def get_data(noise_level=0.05, N_r=200, N_u=20):
    # noise_level = 0.05

    X_r = np.random.uniform(low=-1, high=1, size=(N_r, 1))
    Y_r = np.random.uniform(low=-1, high=1, size=(N_r, 1))


    X_mean = np.mean(X_r)
    X_std = np.std(X_r)

    Y_mean = np.mean(Y_r)
    Y_std = np.std(Y_r)


    X_r = (X_r - X_mean) / X_std
    Y_r = (Y_r - Y_mean) / Y_std

    XY_r = np.hstack((X_r, Y_r))
    XY_r = np.float32(XY_r)

    XYU_u = generate_boundary_data(noise_level, N_u, X_mean, X_std, Y_mean, Y_std)

    N_test = 100
    XY_test = generate_interior_data(N_test, X_mean, X_std, Y_mean, Y_std)

    return XYU_u, XY_r, XY_test, np.float32(X_mean), np.float32(X_std), np.float32(Y_mean), np.float32(Y_std)



'''
XYU = generate_boundary_data(noise_level=0, N_u=50, X_mean=0, X_std=1, Y_mean=0, Y_std=1)
X = XYU[:,0]
Y = XYU[:,1]
U = XYU[:,2]
plt.scatter(Y, U)
plt.show()
'''


