import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten
from custom_layers.bjorcklinear import BjorckLinear
from custom_activations.activations import group_sort


def get_discriminator(input_shape=1, depth=5, width=64, bjorck_beta=0.5, bjorck_iter=5, bjorck_order=2, group_size=2):
    def activation(x):
        # return tf.nn.relu(x)
        return group_sort(x, group_size=group_size)


    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(BjorckLinear(width, activation=activation,
                           bjorck_beta=bjorck_beta, bjorck_iter=bjorck_iter,
                           bjorck_order=bjorck_order))
    for _ in range(depth - 1):
        model.add(BjorckLinear(width, activation=activation,
                               bjorck_beta=bjorck_beta, bjorck_iter=bjorck_iter,
                               bjorck_order=bjorck_order
                               ))
    model.add(BjorckLinear(width, activation=None))

    return model
