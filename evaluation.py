import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, GlobalMaxPool2D
from math import floor

import datetime
import pickle
import os

import wandb
from wandb.keras import WandbCallback

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


def load(training_size=50000):
    '''
    Load raw data and save as artifact
    '''
    with wandb.init(project=project_name, job_type='load-data') as run:
        # Load data from MNIST
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Separate into training and validation
        x_train, x_val = x_train[:training_size], x_train[training_size:]
        y_train, y_val = y_train[:training_size], y_train[training_size:]

        # Register as artifact
        datasets = {'x': x_train, 'y': y_train}, {
            'x': x_val, 'y': y_val}, {'x': x_test, 'y': y_test}
        names = ['training', 'validation', 'testing']

        raw_data = wandb.Artifact(
            'mnist-raw', type='dataset',
            description="Raw MNIST dataset, split into train/val/test",
            metadata={"source": "keras.datasets.mnist",
                      "sizes": [len(dataset) for dataset in datasets]})

        # Create files
        for name, data in zip(names, datasets):
            with raw_data.new_file(name + '.npz', mode='wb') as file:
                x, y = data
                np.savez(file, **data)

        run.log_artifact(raw_data)


def preprocess(steps):
    def extract(dataset, normalize=True, expand_dims=True):
        x, y = dataset

        if normalize:
            x = x / 255.0

        x = x.astype(np.float32)

        if expand_dims:
            x = np.reshape(x, (-1, 28, 28, 1))

        y = np.eye(10)[y]

        return {'x': x, 'y': y}

    def read(data_dir, split):
        filename = split + ".npz"
        with open(os.path.join(data_dir, filename), 'rb') as file:
            npzfile = np.load(file)
            x, y = npzfile['x'], npzfile['y']
            return x, y

    with wandb.init(project=project_name, job_type='preprocess-data') as run:

        # Create artifact
        processed_data = wandb.Artifact(
            'mnist-preprocess', type='dataset',
            description='Preprocessed MINST dataset',
            metadata=steps)

        # Load raw data
        raw_data_artifact = wandb.use_artifact('mnist-raw:latest')
        raw_dataset = raw_data_artifact.download()

        # Save preprocessed data
        for split in ['training', 'validation', 'testing']:
            raw_split = read(raw_dataset, split)
            preprocessed_dataset = extract(raw_split, **steps)

            with processed_data.new_file(split + '.npz', mode='wb') as file:
                np.savez(file, **preprocessed_dataset)

        run.log_artifact(processed_data)


def train(model, training_x, training_y, testing_x, testing_y, name, epoch=1, batch_size=32, lr=1e-3):
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

    wandb.init(project="my-test-project",
               entity="yizhonghu",
               group=name,
               job_type='train',
               config={
                   "model": name,
                   "dataset": "MNIST",
                   "learning_rate": lr,
                   "epochs": epoch,
                   "batch_size": batch_size
               })

    model.fit(x=training_x, y=training_y,
              epochs=epoch, batch_size=batch_size,
              validation_data=(testing_x, testing_y),
              callbacks=[WandbCallback()])

    return model


def create_mlp(hidden_layer_sizes=[16, 16], activation='relu'):
    model = tf.keras.models.Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation=activation))
    model.add(Dense(10, activation='softmax'))

    return model


def train_model(create_fn, exp_name, config):
    with wandb.init(project=project_name, job_type="train-model", group=exp_name, config=config) as run:
        config = wandb.config

        model_config = config['model']
        train_config = config['train']

        # Choose which data to load
        if train_config['dataset'] == 'mnist':
            dataset_artifact = wandb.use_artifact('mnist-preprocess:latest')
        elif train_config['dataset'] == 'mnist-shift':
            dataset_artifact = wandb.use_artifact('mnist-shift:latest')
        else:
            raise ValueError('Incorrect name of dataset')

        dataset_dir = dataset_artifact.download()

        # Load data
        def load_data(name):
            with np.load(os.path.join(dataset_dir, name + '.npz'), 'rb') as file:
                x, y = file['x'], file['y']
                return x, y

        (x_train, y_train), (x_val, y_val), (x_test, y_test) = tuple(load_data(name)
                                                                     for name in ['training', 'validation', 'testing'])

        # Create model and train
        model = create_fn(**model_config)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=train_config['learning_rate']),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc')])
        model.fit(x=x_train, y=y_train,
                  validation_data=(x_val, y_val),
                  epochs=train_config['epochs'],
                  batch_size=train_config['batch_size'],
                  callbacks=[WandbCallback()])

    return model


def sample_shift(config):
    with wandb.init(project=project_name, job_type='sample-shift-data', config=config) as run:
        config = wandb.config

        dataset_shift_artifact = wandb.Artifact(
            'mnist-shift', type='dataset',
            description='Naive Sampled Shifted Data',
            metadata=dict(config))

        dataset_preprocess_artifact = wandb.use_artifact(
            'mnist-preprocess:latest')
        data_preprocess_dir = dataset_preprocess_artifact.download()

        for name in ['training', 'validation', 'testing']:
            file_name = name + '.npz'
            with np.load(os.path.join(data_preprocess_dir, file_name), 'rb') as file:
                x, y = file['x'], file['y']
                num_shift_sample = floor(x.shape[0] * config['sample_rate'])
                x, y = shift_data(
                    x, y, num_shift_sample=num_shift_sample, shift_max=config['shift_max'])
            with dataset_shift_artifact.new_file(file_name, 'wb') as file:
                np.savez(file, x=x, y=y)

        run.log_artifact(dataset_shift_artifact)


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


def accuracy_on_shift(model, max_shift=5):
    '''
    Evaluates the model with different offsets, loads the file if exists one with model_full_name

    Parameter:
        model: keras.Model, the model to evaluate
        max_shift: int, the maximum number of units during evaluation
        overwrite: bool, if override the existing file

    Return:
        a 2D np.ndarray, representing the accuracy after a col and row shift represented by the index
    '''

    x_test_shift = x_shift(x_test, pad_width=max_shift + 1)

    x = np.arange(-max_shift, max_shift + 1, 1)
    y = np.arange(-max_shift, max_shift + 1, 1)

    accuracies = np.array([[model.evaluate(x_test_shift(col, row), y_test)[1]
                            for row in x]
                           for col in y])

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
    save_path = os.path.join(wandb.run.dir, 'acc.html')
    fig.write_html(save_path)
    return save_path


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


def evaluate_model(model, exp_name, config):
    with wandb.init(project=project_name, job_type='evaluate-model', group=exp_name, config=config) as run:
        # Choose which data to load
        if config['dataset'] == 'mnist':
            dataset_artifact = wandb.use_artifact('mnist-preprocess:latest')
        elif config['dataset'] == 'mnist-shift':
            dataset_artifact = wandb.use_artifact('mnist-shift:latest')
        else:
            raise ValueError('Incorrect name of dataset')

        dataset_dir = dataset_artifact.download()

        # Load data
        def load_data(name):
            with np.load(os.path.join(dataset_dir, name + '.npz'), 'rb') as file:
                x, y = file['x'], file['y']
                return x, y

        x_test, y_test = load_data('testing')

        res = model.evaluate(x_test, y_test, callbacks=[])
        run.summary['test_loss'] = res[0]
        run.summary['test_acc'] = res[1]

        accuracies_mlp = accuracy_on_shift(
            model, max_shift=config['max_shift'])
        run.summary['accuracies'] = accuracies_mlp
        if not config['extrapolation']:
            run.summary['MSE'] = mean_squared_error(accuracies_mlp)
        else:
            train_shift = config['train_shift']
            run.summary['MSE'] = mean_squared_error(
                accuracies_mlp[-train_shift:train_shift+1, -train_shift:train_shift+1])
            run.summary['MSE_Xtra'] = mean_squared_error(accuracies_mlp)
        save_path = draw_accuracy(
            accuracies_mlp, 'MLP', max_shift=config['max_shift'])
        with open(save_path) as html:
            wandb.log({'accuracies_on_shift': wandb.Html(html)})

def create_cnn(conv_size=(3, 3), pool_size=(2, 2),
               conv_layers=[20, 50], pool_type='max',
               conv_padding='valid',
               global_pool='none', conv_dropout=0.0,
               hidden_layers=[500], dense_dropout=0.0,
               activation='relu'):
    def conv(model, size, pool=True, input_shape=None):
        if input_shape is None:
            model.add(Conv2D(size, conv_size, activation=activation, padding=conv_padding))
        else:
            model.add(Conv2D(size, conv_size, activation=activation, padding=conv_padding, input_shape=input_shape))
        
        if pool:
            if pool_type == 'max':
                model.add(MaxPool2D(pool_size))
            elif pool_type == 'average':
                model.add(AveragePooling2D(pool_size))

    model = tf.keras.models.Sequential()
    conv(model, conv_layers[0], input_shape=(28, 28, 1))
    if global_pool == 'max':
        for layer in conv_layers[1:-1]:
            conv(model, layer)
        conv(model, conv_layers[-1], pool=False)
        model.add(GlobalMaxPool2D())
    elif global_pool == 'average':
        for layer in conv_layers[1:-1]:
            conv(model, layer)
        conv(model, conv_layers[-1], pool=False)
        model.add(GlobalAveragePooling2D())
    else:
        for layer in conv_layers[1:]:
            conv(model, layer)
        model.add(Flatten())
    model.add(Dropout(conv_dropout))
    
    for layer in hidden_layers:
        model.add(Dense(layer, activation=activation))
    model.add(Dropout(dense_dropout))
    model.add(Dense(10, activation='softmax'))

    return model