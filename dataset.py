import tensorflow as tf
import numpy as np
import wandb
import os

from common import *


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
    '''
    Preprocess data for CNN training
    '''
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


def data_process(inpt, output, description, job_type, project=project_name, compress=False,
                 x_in_keys='x', y_in_keys='y', x_out_keys='x', y_out_keys='y'):
    '''
    Wandb wrapper for data processing. Loads training, validation, and testing data from input
    and process them with the function specified

    Parameters:
        inpt: name of input dataset artifact
        output: name of output dataset artifact
        description: description of output dataset artifact
        job_type: wandb job name
        project: which project to commit to, default: project_name
        compress: if the output should be compressed with np.savez_compress, default: False
        x_in_keys: subarrays of the model inputs to process
        y_in_keys: subarrays of the model outputs to process
        x_out_keys: subarrays of the processed model inputs
        y_out_keys: subarrays of the processed model outputs

    Return:
        a functional wrapper

    Functional input:
        func: a function that takes model inputs and outputs and keyword arguments
            Parameters:
                x: model inputs
                y: model outputs
                **kwargs: keyword arguments for data processing
    Functional output:
        a function that takes the keyword arguments, processes data accordingly, and uploads to wandb
            Parameters:
                config: dict, keyword arguments
    '''
    def functional(func):
        def wrapper(config):
            with wandb.init(project=project, job_type=job_type, config=config) as run:
                output_artifact = wandb.Artifact(
                    output, type='dataset',
                    description=description,
                    metadata=dict(config))

                input_artifact = wandb.use_artifact(inpt)
                input_dir = input_artifact.download()

                for name in ['training', 'validation', 'testing']:
                    file_name = name + '.npz'
                    with np.load(os.path.join(input_dir, file_name), 'rb') as file:
                        x = [file[key] for key in x_in_keys]
                        y = [file[key] for key in y_in_keys]
                        x, y = func(x, y, **config)
                    with output_artifact.new_file(file_name, 'wb') as file:
                        arrays = {**dict(zip(x_out_keys, x)),
                                  **dict(zip(y_out_keys, y))}
                        if compress:
                            np.savez_compressed(file, **arrays)
                        else:
                            np.savez(file, **arrays)

                run.log_artifact(output_artifact)
        return wrapper

    return functional


@data_process('mnist-preprocess:latest', 'mnist-shift', 'Naive Sampled Shifted Data', 'sample-shift-data', compress=True)
def sample_shift(x, y, sample_rate=.1, shift_max=5):
    '''
    Samples shifted data and combines them into one data set

    Parameter:
        x: np.ndarray, inputs to the model
        y: np.ndarray, labels of the inputs
        sample_rate: the ratio of data sampled in each shift, default: 0.1
        shift_max: the maximum shift, default 5

    Return:
        The samples for input and their corresponding labels, in a tuple
    '''
    num_shift_sample = np.floor(x.shape[0] * sample_rate)
    shift = x_shift(x, pad_width=shift_max + 1)
    x_samples = list()
    y_samples = list()
    for row in range(-shift_max, shift_max + 1):
        for col in range(-shift_max, shift_max + 1):
            permu = np.random.choice(
                x.shape[0], num_shift_sample, replace=False)
            x_samples.append(shift(row, col)[permu, :, :, :])
            y_samples.append(tf.gather(y, permu, axis=0))

    x_samples = np.concatenate(x_samples, axis=0)
    y_samples = tf.concat(y_samples, axis=0)

    return x_samples, y_samples


@data_process('mnist-preprocess:latest', 'mnist-pad', 'Padded MNIST dataset', 'pad_data')
def pad_data(x, y, pad_width=10):
    '''
    Pad the input with some width of zero values
    '''
    x = np.pad(x, ((0, 0),
                   (pad_width, pad_width),
                   (pad_width, pad_width),
                   (0, 0)),
               constant_values=((0, 0),) * 4)
    return x, y


@data_process(
    inpt='mnist-pad:latest', output='mnist-pad-loc',
    description='Padded and shifted data with number position',
    job_type='roll_loc', y_out_keys=['class', 'coord'])
def roll_loc_data(x, y, duplicate=1, roll_max=10):
    '''
    For each example, roll it in some random direction and record the roll value with two extra dimensions of output
    '''
    x_samples = list()
    y_coords = list()
    for copy in range(duplicate):
        for x_example, y_example in zip(x, y):
            coords = np.random.randint(-roll_max, roll_max + 1, size=2)
            x_samples.append(np.roll(x_example, coords, axis=(
                0, 1)).reshape((1,) + x_example.shape))
            y_coords.append(coords.reshape(1, -1))

    x_samples = np.concatenate(x_samples, axis=0)
    y_samples = tf.concat([y] * duplicate, axis=0)
    y_coords = np.concatenate(y_coords, axis=0)

    return x_samples, (y_samples, y_coords)
