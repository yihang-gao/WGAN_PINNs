import numpy as np
import tensorflow as tf


def group_sort(x, group_size=2, axis=-1):
    shape = _get_shape_as_list(x)
    units = shape[axis]
    assert isinstance(units, (int, np.int32, np.int64))

    if units % group_size:
        raise ValueError('number of features({}) is not a '
                         'multiple of grouping size({})'.format(units, group_size))

    index = len(shape) if axis == -1 else axis + 1

    new_shape = shape.copy()
    new_shape[axis] = -1
    new_shape.insert(index, group_size)

    ret = tf.reshape(x, new_shape)

    if group_size == 2:
        ret1 = tf.reduce_max(ret, axis=index, keepdims=True)
        ret2 = tf.reduce_min(ret, axis=index, keepdims=True)
        ret = tf.concat([ret1, ret2], axis=index)
    else:
        ret = tf.sort(ret, axis=index)

    ret = tf.reshape(ret, shape)

    return ret


def _get_shape_as_list(x):
    assert isinstance(x, tf.Tensor)
    shape = x.shape.as_list()
    shape_tensor = tf.shape(x)
    ret = []
    for i in range(len(shape)):
        if shape[i] is None:
            ret.append(shape_tensor[i])
        else:
            ret.append(shape[i])

    return ret

