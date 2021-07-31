import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from custom_layers.bjorcklinear import BjorckLinear
from custom_activations.activations import group_sort


def get_encoder(input_shape=1, depth=5, width=64):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(width, input_shape=input_shape, activation='tanh'))
    for _ in range(depth - 1):
        model.add(Dense(width, activation='tanh'))
    model.add(Dense(1, activation='tanh'))

    return model