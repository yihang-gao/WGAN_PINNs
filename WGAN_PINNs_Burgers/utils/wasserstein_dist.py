import ot
import numpy as np
import time
import tensorflow as tf

# def wass1_dis(xs, xt):
#   M = ot.dist(xs, xt, 'euclidean');
'''
t = time.time()
n=10000
xs = np.random.uniform(size=[n, 28 * 28])
xt = np.random.uniform(size=[n, 28 * 28])
M = ot.dist(xs, xt, 'euclidean')
print(time.time()-t)
a, b = np.ones((n,)) / n, np.ones((n,)) / n
W = ot.emd2(a, b, M)
# print(np.shape(a))
'''

'''
n = 10000
t1 = time.time()
xs = tf.random.uniform(shape=[n, 28 * 28])
xt = tf.random.uniform(shape=[n, 28 * 28])
A = tf.math.reduce_sum(tf.math.square(xs), axis=-1, keepdims=True)
B = tf.math.reduce_sum(tf.math.square(xt), axis=-1, keepdims=True)
AB = tf.matmul(xs, xt, transpose_b=True)
M = A - 2 * AB + tf.transpose(B)
print(time.time() - t1)
t2 = time.time()
a, b = np.ones((n,)) / n, np.ones((n,)) / n
W = ot.emd2(a, b, M, numItermax=100000)
print(time.time() - t2)
print(W)
# print(tf.shape(M))
'''


def wasserstein_dist(xs, xt):
    n = np.shape(xs)[0]
    xs = tf.convert_to_tensor(xs, dtype=tf.float32, dtype_hint=None, name=None)
    xt = tf.convert_to_tensor(xt, dtype=tf.float32, dtype_hint=None, name=None)
    A = tf.math.reduce_sum(tf.math.square(xs), axis=-1, keepdims=True)
    B = tf.math.reduce_sum(tf.math.square(xt), axis=-1, keepdims=True)
    AB = tf.matmul(xs, xt, transpose_b=True)
    M = A - 2 * AB + tf.transpose(B)
    M = tf.sqrt(tf.abs(M))
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    W = ot.emd2(a, b, M, numItermax=10000000)
    return W

'''
n = 1000
xs = tf.random.uniform(shape=(n, 2), minval=0, maxval=1)
xt = tf.random.uniform(shape=(n, 2), minval=0, maxval=1)
W1 = wasserstein_dist(xs, xt)
print(W1)
M = ot.dist(xs, xt, 'euclidean')
a, b = np.ones((n,)) / n, np.ones((n,)) / n
W2 = ot.emd2(a, b, M, numItermax=10000000)
print(W2)
'''
