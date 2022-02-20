from evaluation import *

exp_name = 'CNN-pad-1'
dataset_name = 'mnist-pad'
model_config = {
    'input_shape': (48, 48, 1),
    'conv_size': (5, 5),
    'conv_layers': [32, 64, 128],
    'conv_padding': 'valid',
    'pool_size': (2, 2),
    'pool_type': 'max',
    'global_pool': 'average',
    'conv_dropout': 0.0,
    'hidden_layers': [32, 32],
    'dense_dropout': 0.0,
    'activation': 'relu'
}

model_cnn = create_cnn(**model_config)
model_cnn.compile()
model_cnn.summary()


# sample_shift({'sample_rate': 1/10, 'shift_max': 10})