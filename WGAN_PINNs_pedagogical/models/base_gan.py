import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from models.discriminator import get_discriminator
from models.generator import get_generator
from models.encoder import get_encoder
from tqdm import tqdm
from joblib import load
import os
from pathlib import Path
from os import path

# from utils.wasserstein_dist import wasserstein_dist


class BaseGAN(object):

    def __init__(self,
                 noise_level=0.05,
                 N_r=200,
                 N_u=20,
                 X_mean=0,
                 X_std=1,
                 par_pinns=1,
                 z_shape=50,
                 out_dim=1,
                 num_itr=50,
                 g_depth=5,
                 g_width=64,
                 d_depth=5,
                 d_width=64,
                 lrg=1e-4,
                 lrd=1e-4,
                 beta_1=0.9,
                 beta_2=0.999,
                 bjorck_beta=0.5,
                 bjorck_iter=5,
                 bjorck_order=2,
                 group_size=2):

        self.noise_level = noise_level
        self.N_r = N_r
        self.N_u = N_u
        self.X_mean = X_mean
        self.X_std = X_std
        self.z_shape = z_shape
        self.out_dim = out_dim
        self.num_itr = num_itr
        self.Jacobian = 1 / self.X_std
        self.par_pinns = par_pinns
        self.x_shape = 1
        self.k_d = 2
        self.k_g = 5

        self.d_depth = d_depth
        self.d_width = d_width
        self.g_depth = g_depth
        self.g_width = g_width

        # network initialization
        self.G = get_generator(input_shape=(self.z_shape + self.x_shape,), output_shape=self.out_dim, depth=g_depth,
                               width=g_width)
        self.D = get_discriminator(input_shape=(self.out_dim + self.x_shape,), depth=d_depth, width=d_width,
                                   bjorck_beta=bjorck_beta, bjorck_iter=bjorck_iter, bjorck_order=bjorck_order,
                                   group_size=group_size)

        self.G_optimizer = Adam(learning_rate=lrg, beta_1=beta_1, beta_2=beta_2)
        self.D_optimizer = Adam(learning_rate=lrd, beta_1=beta_1, beta_2=beta_2)
        # self.G_optimizer = RMSprop(learning_rate=lrg)
        # self.D_optimizer = RMSprop(learning_rate=lrd)
        self.Loss = 0.0

    def f(self, X_normalized):
        X = X_normalized * self.X_std + self.X_mean
        return - np.pi ** 2 * tf.sin(np.pi * X) - np.pi * tf.cos(np.pi * X) * tf.sin(np.pi * X) ** 2

    def generator_loss(self, fake_output, residual):
        return tf.math.reduce_mean(fake_output) + self.par_pinns * residual

    def discriminator_loss(self, real_output, fake_output):
        return -tf.math.reduce_mean(fake_output) + tf.math.reduce_mean(real_output)

    def get_r(self, X_r, noises):
        with tf.GradientTape(watch_accessed_variables=False) as pde_tape2:
            pde_tape2.watch(X_r)
            with tf.GradientTape(watch_accessed_variables=False) as pde_tape1:
                pde_tape1.watch(X_r)
                u = self.G(tf.concat([noises, X_r], axis=1), training=True)
            u_x = pde_tape1.gradient(u, X_r)

        u_xx = pde_tape2.gradient(u_x, X_r)
        f = self.f(X_r)
        r = (self.Jacobian ** 2) * u_xx - (self.Jacobian) * (u ** 2) * u_x - f
        r2 = r ** 2
        return tf.math.reduce_mean(r2)

    @tf.function()
    def train_step_discriminator(self, X_u, XY_u, noises_u):
        with tf.GradientTape() as disc_tape:
            generated_Y = self.G(tf.concat([noises_u, X_u], axis=1), training=False)

            real_output = self.D(XY_u, training=True)
            fake_output = self.D(tf.concat([X_u, generated_Y], axis=1), training=True)

            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.D.trainable_variables)

        self.D_optimizer.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))

    @tf.function()
    def train_step_generator(self, X_u, X_r, noises_u, noises_r):
        with tf.GradientTape(persistent=True) as gen_tape:
            generated_Y = self.G(tf.concat([noises_u, X_u], axis=1), training=True)

            fake_output = self.D(tf.concat([X_u, generated_Y], axis=1), training=False)

            residual = self.get_r(X_r, noises_r)

            gen_loss = self.generator_loss(fake_output, residual)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.G.trainable_variables)

        self.G_optimizer.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables))

    def generate_sample(self, X):
        num = X.shape[0]
        noise = tf.random.normal([num, self.z_shape])
        return self.G(tf.concat([noise, X], axis=1), training=False)


    def get_solution(self, X_normalized):
        X = X_normalized * self.X_std + self.X_mean
        return np.sin(np.pi * X)

    def get_relative_error(self, X, Y, l2, num_test):
        samples = np.zeros((num_test, 2000), dtype=np.float32)
        for i in range(0, 2000):
            samples[:, i:i + 1] = self.generate_sample(X)

        Y_predicted = np.mean(samples, axis=1, dtype=np.float32)[:, None]

        return np.sqrt(tf.reduce_mean(tf.square(Y_predicted - Y)) / l2)

    def train(self, XY_u, X_r, X_test):
        print('--------------Begin Training-----------------')
        count = 0
        X = np.vstack((np.ones(shape=(500, 1)) * 1.0, np.ones(shape=(500, 1)) * (-1.0)))
        X = (X - self.X_mean) / self.X_std
        X = np.float32(X)
        num_test = X_test.shape[0]
        X_u = XY_u[:, 0][:, None]
        num = X_u.shape[0]
        X_u = tf.convert_to_tensor(X_u)
        XY_u = tf.convert_to_tensor(XY_u)
        X_test = tf.convert_to_tensor(X_test)
        Y_test = self.get_solution(X_test)
        l2 = tf.reduce_mean(tf.square(Y_test))
        X_r = tf.convert_to_tensor(X_r)
        time1 = time.time()
        for itr in range(self.num_itr):
            noises_u = tf.random.normal([2 * self.N_u, self.z_shape])
            noises_r = tf.random.normal([self.N_r, self.z_shape])
            for i in range(self.k_d):
                self.train_step_discriminator(X_u, XY_u, noises_u)

            for j in range(self.k_g):
                self.train_step_generator(X_u, X_r, noises_u, noises_r)

            if (itr + 1) % 2000 == 0:
                noises_u = tf.random.normal([1000, self.z_shape])
                noises_test = tf.random.normal([num_test, self.z_shape])
                r2_loss = self.get_r(X_test, noises_test)
                rel_error = self.get_relative_error(X_test, Y_test, l2, num_test)
                u_predict = self.G(tf.concat([noises_u, X], axis=1), training=False)
                w_dis = tf.math.reduce_mean(self.D(tf.concat([X, u_predict], axis=1), training=False))
                print(
                    "itr {}, rel_error is {:e} r2_loss is {:4f}, W loss is {:5f}; Time: {:4f}.\n".format(
                        itr + 1, rel_error, r2_loss, w_dis,
                        time.time() - time1))


                time1 = time.time()



        samples = np.zeros((201, 2000), dtype=np.float32)
        X = np.linspace(-1, 1, 201)[:, None]
        X = (X-self.X_mean)/self.X_std
        for i in range(0, 2000):
            samples[:, i:i + 1] = self.generate_sample(X)

        np.save("data_{:.2f}".format(self.noise_level), samples)
        print('--------------End Training-----------------')
