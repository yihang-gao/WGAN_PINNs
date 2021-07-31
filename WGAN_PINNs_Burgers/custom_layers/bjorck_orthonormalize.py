import tensorflow as tf


@tf.function()
def bjorck_orthonormalize(w, beta=0.5, iters=10, order=1):
    if order == 1:
        for _ in tf.range(iters):
            wtw = tf.linalg.matmul(w, w, transpose_a=True, transpose_b=False)
            w = (1 + beta) * w - beta * tf.linalg.matmul(w, wtw)
    elif order == 2:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)
        for _ in tf.range(iters):
            wtw = tf.linalg.matmul(w, w, transpose_a=True, transpose_b=False)
            w_wtw = tf.linalg.matmul(w, wtw)
            w_wtw2 = tf.linalg.matmul(w_wtw, wtw)
            # wtw2 = tf.linalg.matmul(wtw, wtw)
            w = (+ (15 / 8) * w
                 - (5 / 4) * w_wtw
                 + (3 / 8) * w_wtw2)
    elif order == 3:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)
        for _ in tf.range(iters):
            wtw = tf.linalg.matmul(w, w, transpose_a=True, transpose_b=False)
            w_wtw = tf.linalg.matmul(w, wtw)
            w_wtw2 = tf.linalg.matmul(w_wtw, wtw)
            w_wtw3 = tf.linalg.matmul(w_wtw2, wtw)

            w = (+ (35 / 16) * w
                 - (35 / 16) * w_wtw
                 + (21 / 16) * w_wtw2
                 - (5 / 16) * w_wtw3)
    elif order == 4:
        if beta != 0.5:
            print("Bjorck orthonormalization with order more than 1 requires a beta of 0.5. ")
            exit(-1)

        for _ in tf.range(iters):
            wtw = tf.linalg.matmul(w, w, transpose_a=True, transpose_b=False)
            w_wtw = tf.linalg.matmul(w, wtw)
            w_wtw2 = tf.linalg.matmul(w_wtw, wtw)
            w_wtw3 = tf.linalg.matmul(w_wtw2, wtw)
            w_wtw4 = tf.linalg.matmul(w_wtw3, wtw)

            w = (+ (315 / 128) * w
                 - (105 / 32) * w_wtw
                 + (189 / 64) * w_wtw2
                 - (45 / 32) * w_wtw3
                 + (35 / 128) * w_wtw4)

    else:
        print("The requested order for orthonormalization is not supported. ")
        exit(-1)

    return w
