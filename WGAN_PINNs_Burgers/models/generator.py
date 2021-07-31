import tensorflow as tf
# from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
import numpy as np


def get_generator(input_shape=(1,), output_shape=(1, ), depth=5, width=64):
    model = Sequential()
    model.add(Dense(width, input_shape=input_shape, activation='tanh'))
    for _ in range(depth - 1):
        model.add(Dense(width, activation='tanh'))
    model.add(Dense(np.prod(output_shape, dtype=int), activation=None))
    # model.add(Reshape(output_shape))

    return model
