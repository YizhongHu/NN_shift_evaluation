import numpy as np
import tensorflow as tf

project_name = 'mnist-shift'

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

x_train, x_val = x_train[:50000], x_train[50000:]
y_train, y_val = y_train[:50000], y_train[50000:]

def x_shift(x, pad_width=10):
    '''
    Returns a function that generates shift of input padded by 0

    Parameters:
        x: array-like, with 4 dimensions, with the first representing the batch size
        pad_width: int, the padding given to the image

    Return:
        A function that shifts the input x
        Parameters: 
            col: the number of shifts in the +hor direction
            row: the number of shifts in the +ver direction

        Return:
            The input shifted
    '''
    x_padded = np.pad(x, ((0, 0), (pad_width, pad_width),
                      (pad_width, pad_width), (0, 0)), constant_values=((0, 0),) * 4)

    def _shift(row, col):
        return x_padded[:, pad_width - row:-pad_width - row, pad_width - col:-pad_width - col, :]

    return _shift
