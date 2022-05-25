import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dense, \
    Dropout, GlobalAveragePooling2D, GlobalMaxPool2D, ZeroPadding2D, Reshape, Activation
from math import floor

import datetime
import pickle
import os

import wandb
from wandb.keras import WandbCallback

from .common import *
from .metrics import FalseNegative, FalsePositive, CountingError, DuplicateOmission, DuplicateError, METRICS_DICT


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


def create_mlp(hidden_layer_sizes=[16, 16], dropout=0.0, activation='relu'):
    '''
    Create an mlp with the specified hidden layers. Dropout happens before final layer
    '''
    model = tf.keras.models.Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax'))

    return model


def train_model(create_fn, exp_name, config):
    '''
    wandb wrapper for model.fit

    Parameters:
        create_fn: The function that creates the model
        exp_name: The group name of the experiments
        config: The model and training parameters configuration

    Return:
        Trained model
    '''
    with wandb.init(project=project_name, job_type="train-model", group=exp_name, config=config) as run:
        config = wandb.config

        model_config = config['model']
        train_config = config['train']

        # Choose which data to load
        if train_config['dataset'] == 'mnist':
            dataset_artifact = wandb.use_artifact('mnist-preprocess:latest')
        elif train_config['dataset'] == 'mnist-shift':
            dataset_artifact = wandb.use_artifact('mnist-shift:training')
        elif train_config['dataset'] == 'mnist-pad':
            dataset_artifact = wandb.use_artifact('mnist-pad:latest')
        elif train_config['dataset'] == 'mnist-shift-pad':
            dataset_artifact = wandb.use_artifact('mnist-shift-pad:latest')
        elif train_config['dataset'] == 'mnist-multiple':
            dataset_artifact = wandb.use_artifact('mnist-multiple:latest')
        else:
            raise ValueError('Incorrect name of dataset')

        dataset_dir = dataset_artifact.download()

        # Load data
        def load_data(name):
            with np.load(os.path.join(dataset_dir, name + '.npz'), 'rb') as file:
                x, y = file['x'], file['y']
                return x, y

        (x_trn, y_trn), (x_val, y_val) = tuple(load_data(name)
                                               for name in ['training', 'validation'])

        # Create model and train
        model = create_fn(**model_config)
        if train_config['metric'] == 'accuracy':
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=train_config['learning_rate']),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc')])
            
        elif train_config['metric'] == 'counting':
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=train_config['learning_rate']),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[CountingError(name='err'),
                         FalseNegative(name='fn'),
                         FalsePositive(name='fp'),
                         DuplicateOmission(name='omit'),
                         DuplicateError(name='dup_err')])
        else:
            pass
        
        model.summary()
        model.fit(x=x_trn, y=y_trn,
                  validation_data=(x_val, y_val),
                  epochs=train_config['epochs'],
                  batch_size=train_config['batch_size'],
                  callbacks=[WandbCallback()])

    return model


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

    accuracies = np.array([[model.evaluate(x_test_shift(row, col), y_test)[1]
                            for col in y]
                           for row in x])

    return accuracies


def accuracy_on_roll(model, data_x, data_y, max_shift=10):
    x = np.arange(-max_shift, max_shift + 1, 1)
    y = np.arange(-max_shift, max_shift + 1, 1)

    accuracies = np.array([[model.evaluate(np.roll(data_x, (x_roll, y_roll), axis=(1, 2)), data_y)[1]
                            for y_roll in y]
                           for x_roll in x])

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
    xm, ym = np.meshgrid(x, y, indexing='ij')

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


def evaluate_model(model, exp_name, config):
    '''
    Evaluate the trained model

    Parameters:
        model: The trained model
        exp_name: The group name of the experiment
        config: evaluation configurations
    '''
    with wandb.init(project=project_name, job_type='evaluate-model', group=exp_name, config=config) as run:
        # Choose which data to load
        if config['dataset'] == 'mnist':
            dataset_artifact = wandb.use_artifact('mnist-preprocess:latest')
        elif config['dataset'] == 'mnist-shift':
            dataset_artifact = wandb.use_artifact('mnist-shift:training')
        elif config['dataset'] == 'mnist-pad':
            dataset_artifact = wandb.use_artifact('mnist-pad:latest')
        elif config['dataset'] == 'mnist-shift-pad':
            dataset_artifact = wandb.use_artifact('mnist-shift-pad:latest')
        elif config['dataset'] == 'mnist-multiple':
            dataset_artifact = wandb.use_artifact('mnist-multiple:latest')
        else:
            raise ValueError('Incorrect name of dataset')

        dataset_dir = dataset_artifact.download()

        # Load data
        def load_data(name):
            with np.load(os.path.join(dataset_dir, name + '.npz'), 'rb') as file:
                x, y = file['x'], file['y']
                return x, y

        x_t, y_t = load_data('testing')

        # Evaluate with the loaded data
        res = model.evaluate(x_t, y_t, callbacks=[])
        for value, metric in zip(res, model.metrics):
            run.summary['test_' + metric.name] = value
        

        # Check the accuracies on shift
        if config['dataset'] in {'mnist', 'mnist-shift'}:
            accuracies_mlp = accuracy_on_shift(
                model, max_shift=config['max_shift'])
        elif config['dataset'] in {'mnist-pad', 'mnist-shift-pad'}:
            # Load unshifted data
            dataset_artifact = wandb.use_artifact('mnist-pad:latest')
            dataset_dir = dataset_artifact.download()
            x_t, y_t = load_data('testing')
            # Evaluate Shifted Accuracies
            accuracies_mlp = accuracy_on_roll(
                model, x_t, y_t, max_shift=config['max_shift'])
        elif config['dataset'] == 'mnist-multiple':
            return

        # Record run results
        run.summary['accuracies'] = accuracies_mlp
        if not config['extrapolation']:
            run.summary['MSE'] = mean_squared_error(accuracies_mlp)
        else:
            train_shift = config['train_shift']
            shape = accuracies_mlp.shape
            center_x, center_y = shape[0]//2, shape[1]//2
            run.summary['MSE'] = mean_squared_error(
                accuracies_mlp[center_x-train_shift:center_x+train_shift+1, center_y-train_shift:center_y+train_shift+1])
            run.summary['MSE_Xtra'] = mean_squared_error(accuracies_mlp)
        run.summary['min_acc'] = np.amin(accuracies_mlp)
        run.summary['max_acc'] = np.amax(accuracies_mlp)
        run.summary['10%_acc'] = np.percentile(accuracies_mlp, 10)
        run.summary['90%_acc'] = np.percentile(accuracies_mlp, 90)
        run.summary['acc_std'] = np.std(accuracies_mlp)
        save_path = draw_accuracy(
            accuracies_mlp, exp_name, max_shift=config['max_shift'])
        with open(save_path) as html:
            wandb.log({'accuracies_on_shift': wandb.Html(html)})


def create_cnn(input_shape=(28, 28, 1),
               conv_size=(3, 3), pool_size=(2, 2),
               conv_layers=[20, 50], pool_type='max',
               conv_padding='valid', conv_bias=True,
               global_pool='none', conv_dropout=0.0,
               hidden_layers=[500], dense_dropout=0.0,
               activation='relu', dense_bias=True,
               output_shape=10, output_activation='softmax'):
    '''
    Creates a CNN
    '''
    def conv(model, size, pool=True, input_shape=None):
        '''
        Creates a convolution layer
        '''
        if conv_padding in {'valid', 'same'}:
            model.add(Conv2D(size, conv_size,
                             activation=activation, padding=conv_padding, use_bias=conv_bias))
        elif conv_padding == 'lossless':
            model.add(ZeroPadding2D((conv_size[0] - 1, conv_size[1] - 1)))
            model.add(Conv2D(size, conv_size,
                             activation=activation, padding='valid', use_bias=conv_bias))

        if pool:
            if pool_type == 'max':
                model.add(MaxPool2D(pool_size))
            elif pool_type == 'average':
                model.add(AveragePooling2D(pool_size))
            else:
                # No Pooling
                pass

    # Convolution
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    if len(conv_layers) > 1:
        # Convolution hidden layers
        for layer in conv_layers[0:-1]:
            conv(model, layer)

    # Flattening layer
    if global_pool == 'max':
        conv(model, conv_layers[-1], pool=False)
        model.add(GlobalMaxPool2D())
    elif global_pool == 'average':
        conv(model, conv_layers[-1], pool=False)
        model.add(GlobalAveragePooling2D())
    else:
        conv(model, conv_layers[-1], pool=True)
        model.add(Flatten())

    model.add(Dropout(conv_dropout))

    # MLP
    for layer in hidden_layers:
        model.add(Dense(layer, use_bias=dense_bias, activation=activation))
    model.add(Dropout(dense_dropout))
    if output_activation == 'count_prob':
        model.add(Dense(output_shape[0] * output_shape[1], use_bias=dense_bias))
        model.add(Reshape(output_shape))
        model.add(Activation(tf.keras.activations.softmax))
    else:
        model.add(Dense(output_shape, use_bias=dense_bias,
                  activation=output_activation))

    return model


def load_model(id, custom_objects=None):
    '''
    Load the model with the given training run id
    '''
    with wandb.init(project=project_name, id=id, resume=True) as run:
        model = tf.keras.models.load_model(wandb.restore('model-best.h5').name, custom_objects=custom_objects)
        # run_info = {'group': run.group}
        return model


def top_k_evaluation(model, exp_name, config):
    '''
    Selects top k loss results from all shifted data
    '''
    with wandb.init(project=project_name,
                    job_type='top-k',
                    group=exp_name,
                    config=config) as run:
        # Choose which data to load
        if config['dataset'] == 'mnist':
            dataset_artifact = wandb.use_artifact('mnist-preprocess:latest')
        elif config['dataset'] == 'mnist-shift':
            dataset_artifact = wandb.use_artifact('mnist-shift:find-error')
        else:
            raise ValueError('Incorrect name of dataset')

        dataset_dir = dataset_artifact.download()

        # Load data
        def load_data(name):
            with np.load(os.path.join(dataset_dir, name + '.npz'), 'rb') as file:
                x, y = file['x'], file['y']
                return x, y

        x_test, y_test = load_data('testing')

        # Calculate loss
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)
        y_test = tf.math.argmax(y_test, axis=1)
        y_pred = model.predict(x_test)
        loss = loss_fn(y_test, y_pred)

        # Pick the highest loss
        top_k_val, top_k_ind = tf.math.top_k(loss, k=config['k'])
        y_test = tf.gather(y_test, top_k_ind)
        y_pred = tf.gather(y_pred, top_k_ind)
        y_pred = tf.math.argmax(y_pred, axis=1)
        x_test = tf.gather(x_test, top_k_ind) * 255
        wandb.log({'top-k-error': [wandb.Image(
            image, mode='L', caption=f'pred: {pred}, label: {label}, loss: {loss}')
            for image, pred, label, loss in zip(x_test, y_pred, y_test, top_k_val)]})


if __name__ == "__main__":
    exp_name = ''
    dataset_name = 'mnist-multiple'
    model_config = {
        'input_shape': (120, 120, 1),
        'conv_size': (7, 7),
        'conv_layers': [16, 16],
        'conv_padding': 'lossless',
        'conv_bias': False,
        'pool_size': (2, 2),
        'pool_type': 'none',
        'global_pool': 'max',
        'conv_dropout': 0.0,
        'hidden_layers': [32],
        'dense_dropout': 0.0,
        'dense_bias': True,
        'activation': 'relu',
        'output_shape': [10, 3],
        'output_activation': 'count_prob'
    }
    config = {
        "model": model_config,
        "train": {
            "model": 'CNN',
            "dataset": dataset_name,
            "learning_rate": 1e-3,
            "epochs": 40,
            "batch_size": 32,
            "metric": 'counting'
        }}
    model_cnn = create_cnn(**model_config)
    train_model(create_cnn, exp_name, config)
    