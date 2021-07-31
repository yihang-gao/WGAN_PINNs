import tensorflow as tf
from custom_layers.bjorck_orthonormalize import bjorck_orthonormalize

# a = tf.constant([[1,2,3],[4,5,6],[2,8,2]], dtype=tf.float32)/4
# a = tf.transpose(a)
a = tf.random.uniform([1000, 1000])
scaling = tf.math.reduce_max(tf.abs(a), axis=None)
print(a)

a = a / scaling /1000.0
print(a)
print(tf.math.reduce_max(tf.abs(a), axis=None))

b = bjorck_orthonormalize(a, beta=0.5, iters=5, order=3)
print(b)

print(tf.matmul(b, b, transpose_a=True))
