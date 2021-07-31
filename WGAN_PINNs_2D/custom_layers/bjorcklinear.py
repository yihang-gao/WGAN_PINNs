import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec
# from tensorflow.python.keras.layers.ops import core as core_ops
import custom_layers.ops_dense as core_ops

from custom_layers.bjorck_orthonormalize import bjorck_orthonormalize


# from custom_activations.activations import group_sort


class BjorckLinear(Layer):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 scaling=True,
                 config=None,
                 bjorck_beta=0.5,
                 bjorck_iter=5,
                 bjorck_order=2,
                 **kwargs):
        super(BjorckLinear, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.scaling = scaling
        self.config = config
        self.bjorck_beta = bjorck_beta
        self.bjorck_iter = bjorck_iter
        self.bjorck_order = bjorck_order

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, **kwargs):
        if self.scaling:
            bjorckscaling = self.get_safe_bjorck_scaling()
            bjorckscaling = bjorckscaling * tf.math.reduce_max(tf.abs(self.kernel), axis=None)
        else:
            bjorckscaling = 1.0

        ortho_w = bjorck_orthonormalize(self.kernel / bjorckscaling,
                                        beta=self.bjorck_beta,
                                        iters=self.bjorck_iter,
                                        order=self.bjorck_order)

        # ortho_w_t = tf.transpose(ortho_w)

        return core_ops.dense(
            inputs,
            ortho_w,
            self.bias,
            self.activation,
            dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = super(BjorckLinear, self).get_config()
        config.update({
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        })
        return config

    def get_safe_bjorck_scaling(self):
        shape = tf.shape(self.kernel)
        shape = tf.cast(shape, dtype=tf.float32)
        m, n = shape[0], shape[1]
        return tf.sqrt(m * n)


'''
@tf.function
def get_safe_bjorck_scaling(weight):
    shape =  tf.shape(weight)
    m, n = shape[0], shape[1]
    return tf.sqrt(m*n)

'''
