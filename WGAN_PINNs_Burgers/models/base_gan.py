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

from utils.wasserstein_dist import wasserstein_dist
from utils.loaddata import generate_boundary_data, generate_interior_data


class BaseGAN(object):

    def __init__(self,
                 noise_level=0.05,
                 N_r=200,
                 N_u=20,
                 X_mean=0,
                 X_std=1,
                 T_mean=1,
                 T_std=1,
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
        self.T_mean = T_mean
        self.T_std = T_std
        self.z_shape = z_shape
        self.out_dim = out_dim
        self.num_itr = num_itr
        self.JacobianX = 1 / self.X_std
        self.JacobianT = 1 / self.T_std
        self.par_pinns = par_pinns
        self.x_shape = 2
        self.k_d = 1
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
        return 0.0

    def generator_loss(self, fake_output, residual):
        return tf.math.reduce_mean(fake_output) + self.par_pinns * residual

    def discriminator_loss(self, real_output, fake_output):
        return -tf.math.reduce_mean(fake_output) + tf.math.reduce_mean(real_output)

    def get_r(self, XT_r, noises):
        X_r = XT_r[:, 0][:, None]
        T_r = XT_r[:, 1][:, None]
        with tf.GradientTape(watch_accessed_variables=False) as pde_tape2:
            pde_tape2.watch(X_r)
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as pde_tape1:
                pde_tape1.watch([X_r, T_r])
                u = self.G(tf.concat([noises, X_r, T_r], axis=1), training=True)
            u_x = pde_tape1.gradient(u, X_r)
            u_t = pde_tape1.gradient(u, T_r)
        u_xx = pde_tape2.gradient(u_x, X_r)
        f = self.f(X_r)
        r = u_t * self.JacobianT + u * u_x * self.JacobianX - 0.01 / np.pi * u_xx * (self.JacobianX ** 2)
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


    def train(self, XTY_u, XT_r, XT_test):
        print('--------------Begin Training-----------------')
        num = 1000
        XTY = generate_boundary_data(noise_level=self.noise_level, N_u=num, X_mean=self.X_mean, X_std=self.X_std,
                                     T_mean=self.T_mean, T_std=self.T_std)
        XT = XTY[:, 0:2]
        num_test = XT_test.shape[0]
        XT_u = XTY_u[:, 0:2]
        XT_u = tf.convert_to_tensor(XT_u)
        XTY_u = tf.convert_to_tensor(XTY_u)
        XT_test = tf.convert_to_tensor(XT_test)
        XT_r = tf.convert_to_tensor(XT_r)
        time1 = time.time()
        for itr in range(self.num_itr):
            noises_u = tf.random.normal([2 * self.N_u, self.z_shape])
            noises_r = tf.random.normal([self.N_r, self.z_shape])
            for i in range(self.k_d):
                self.train_step_discriminator(XT_u, XTY_u, noises_u)

            for j in range(self.k_g):
                self.train_step_generator(XT_u, XT_r, noises_u, noises_r)

            if (itr + 1) % 5000 == 0:
                noises_u = tf.random.normal([2 * num, self.z_shape])
                noises_test = tf.random.normal([num_test, self.z_shape])
                r2_loss = self.get_r(XT_test, noises_test)
                u_predict = self.G(tf.concat([noises_u, XT], axis=1), training=False)
                w_dis = tf.math.reduce_mean(
                    self.D(tf.concat([XT, u_predict], axis=1), training=False)) - tf.math.reduce_mean(
                    self.D(XTY, training=False))
                print(
                    "itr {}, r2_loss is {:4f}, W loss is {:5f}; Time: {:4f}.\n".format(
                        itr + 1, r2_loss, w_dis,
                        time.time() - time1))
                time1 = time.time()


        X_test = (np.linspace(-1, 1, 256) - self.X_mean) / self.X_std
        T_test = (np.linspace(0, 0.75, 4) - self.T_mean) / self.T_std

        X_test = np.broadcast_to(X_test[None, :, None], (4, 256, 1))
        T_test = np.broadcast_to(T_test[:, None, None], (4, 256, 1))
        XT_test = np.concatenate((X_test, T_test), axis=-1)
        XT_test = np.reshape(XT_test, newshape=(-1, 2))

        samples = np.zeros((4 * 256, 2000), dtype=np.float32)
        for i in range(0, 2000):
            samples[:, i:i + 1] = self.generate_sample(XT_test)

        np.save("data_{:.2f}".format(self.noise_level), samples)

        print('--------------End Training-----------------')
