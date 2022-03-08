from evaluation import *


# exp_name = 'CNN-9'
# dataset_name = 'mnist'
# model_config = {
#     'input_shape': (28, 28, 1),
#     'conv_size': (5, 5),
#     'conv_layers': [32, 64],
#     'conv_padding': 'valid',
#     'conv_bias': True,
#     'pool_size': (2, 2),
#     'pool_type': 'average',
#     'global_pool': 'max',
#     'conv_dropout': 0.0,
#     'hidden_layers': [32, 32],
#     'dense_dropout': 0.0,
#     'dense_bias': True,
#     'activation': 'relu'
# }

# model_cnn = create_cnn(**model_config)
# model_cnn.compile()
# model_cnn.summary()

def pad_shift_data(config):
    '''
    Pad the data with 0 pixels, with wandb wrapper
    '''
    def pad(dataset, pad_width):
        x, y = dataset
        x = np.pad(x, ((0, 0),
                       (pad_width, pad_width),
                       (pad_width, pad_width),
                       (0, 0)),
                   constant_values=((0, 0),) * 4)
        return {'x': x, 'y': y}

    def read(data_dir, split):
        filename = split + ".npz"
        with open(os.path.join(data_dir, filename), 'rb') as file:
            npzfile = np.load(file)
            x, y = npzfile['x'], npzfile['y']
            return x, y

    with wandb.init(project=project_name, job_type='pad_data', config=config) as run:
        pad_width = config['pad_width']

        dataset_artifact = wandb.use_artifact('mnist-shift:latest')
        dataset = dataset_artifact.download()

        pad_data_artifact = wandb.Artifact(
            'mnist-shift-pad', type='dataset',
            description="Padded Shifted MNIST dataset",
            metadata=config)

        # Save preprocessed data
        for split in ['training', 'validation', 'testing']:
            raw_split = read(dataset, split)
            padded_dataset = pad(raw_split, pad_width)

            with pad_data_artifact.new_file(split + '.npz', mode='wb') as file:
                np.savez_compressed(file, **padded_dataset)

        run.log_artifact(pad_data_artifact)

config = {
    'pad_width': 10
}
pad_shift_data(config)

# sample_shift({'sample_rate': 1/10, 'shift_max': 10})
