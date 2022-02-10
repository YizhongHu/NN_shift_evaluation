from sklearn.model_selection import learning_curve
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime
import pickle
import os


# Loading in and preprocessing the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # normalizing data
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

x_train = np.reshape(x_train, (-1, 28, 28, 1))
x_test = np.reshape(x_test, (-1, 28, 28, 1))

y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)


def train(model, training_x, training_y, testing_x, testing_y, name,
          epoch=1, batch_size=32, lr=1e-3, model_path='/content/drive/MyDrive/models/mnist/'):
    '''
    Train a model, wrapper for model.fit

    Parameters:
        model: keras.Model, the model to train, the model must be compiled with model.compile
        training_x: array like, the training input
        training_y: array like, the training label
        testing_x: array like, the training input
        testing_y: array like, the training label
        name: str, the name of the model
        epoch: int, the number of epochs to train, default 1
        batch_size: int, the batch size, default 32
        lr: float, the learning rate, default 1e-3, overwrites existing learning rate
        model_path: str, the path to save the model, saves to google drive by default

    Return:
        A string, the full name of the model
    '''
    model.optimizer.learning_rate.assign(lr)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_full_name = model_path + name + '/' + current_time
    model_save_dir = model_full_name + '.ckpt'
    hist_save_dir = model_full_name + '.hist'

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_dir,
                                                     save_weights_only=True,
                                                     verbose=1)
    history = model.fit(x=training_x, y=training_y,
                        epochs=epoch, batch_size=batch_size,
                        validation_data=(testing_x, testing_y),
                        callbacks=[cp_callback])

    with open(hist_save_dir, 'w+') as file:
        pickle.dump(history, file)
        print(f'History dumped to {hist_save_dir}')

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(history.history['acc'])
    ax2.plot(history.history['loss'])
    fig.show()

    return model_full_name


def init_model(model, lr=1e-3):
    '''
    Compile model, wrapper for model.compile

    Parameter:
        model: keras.Model, the model to compile
        lr: float, the learning rate, default: 1e-3 

    Return:
        initialized model
    '''
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc')])
    model.summary()

    return model


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

    def _shift(col, row):
        return x_padded[:, pad_width - row:-pad_width - row, pad_width - col:-pad_width - col, :]

    return _shift


def accuracy_on_shift(model, model_full_name, max_shift=5):
    '''
    Evaluates the model with different offsets, loads the file if exists one with model_full_name

    Parameter:
        model: keras.Model, the model to evaluate
        max_shift: int, the maximum number of units during evaluation

    Return:
        a 2D np.ndarray, representing the accuracy after a col and row shift represented by the index
    '''
    acc_dump_name = model_full_name + '_' + str(max_shift) + '.acc'

    if os.path.isfile():
        with open(acc_dump_name, 'r') as file:
            print(f'Accuracies loaded from {acc_dump_name}')
            return pickle.load(file)

    x_test_shift = x_shift(x_test, pad_width=max_shift + 1)

    x = np.arange(-max_shift, max_shift + 1, 1)
    y = np.arange(-max_shift, max_shift + 1, 1)

    accuracies = np.array([[model.evaluate(x_test_shift(col, row), y_test)[1]
                            for row in x]
                           for col in y])

    with open(acc_dump_name, 'r') as file:
        pickle.dump(accuracies, file)
        print(f'Accuracies dumped into {acc_dump_name}')

    return accuracies


def draw_accuracy(accuracies, name, max_shift=5):
    '''
    Visualize the accuracies at different offsets

    Parameters:
        accuracies: np.ndarray (2*max_shift+1, 2*max_shift+1)
        name: str, the name of the graph
        max_shift: the size of the accuracy of the array
    '''
    x = np.arange(-max_shift, max_shift + 1, 1)
    y = np.arange(-max_shift, max_shift + 1, 1)
    xm, ym = np.meshgrid(x, y)

    fig = go.Figure(data=[go.Surface(x=xm, y=ym, z=accuracies)])
    fig.update_layout(title=name + ' loss at different offset levels',
                      xaxis_title="X offset",
                      yaxis_title="Y offset")
    fig.show()


def mean_squared_error(accuracies):
    '''
    Calculates mean squared error

    Parameters:
        accuracies: np.ndarray

    Return:
        Mean Squared Error
    '''
    return np.mean(np.power(1 - accuracies, 2))


def shift_data(x, y, num_shift_sample=6000, shift_max=5):
    '''
    Samples shifted data and combines them into one data set

    Parameter:
        x: np.ndarray, inputs to the model
        y: np.ndarray, labels of the inputs
        num_shift_samples: int, number of samples taken from each shifted state, default 6000
        shift_max: the maximum shift, default 5

    Return:
        The samples for input and their corresponding labels, in a tuple
    '''
    shift = x_shift(x, pad_width=shift_max + 1)
    x_samples = list()
    y_samples = list()
    for row in range(-shift_max, shift_max + 1):
        for col in range(-shift_max, shift_max + 1):
            permu = np.random.choice(
                x.shape[0], num_shift_sample, replace=False)
            x_samples.append(shift(col, row)[permu, :, :, :])
            y_samples.append(tf.gather(y, permu, axis=0))

    x_samples = np.concatenate(x_samples, axis=0)
    y_samples = tf.concat(y_samples, axis=0)

    return x_samples, y_samples
