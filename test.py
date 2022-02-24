from evaluation import *

exp_name = 'hyp6-ll1-3x3-2L'
dataset_name = 'mnist-pad'
model_config = {
    'input_shape': (48, 48, 1),
    'conv_size': (3, 3),
    'conv_layers': [32, 64],
    'conv_padding': 'lossless',
    'conv_bias': True,
    'pool_size': None,
    'pool_type': 'none',
    'global_pool': 'max',
    'conv_dropout': 0.0,
    'hidden_layers': [32, 32],
    'dense_dropout': 0.0,
    'dense_bias': True,
    'activation': 'relu'
}

model_cnn = create_cnn(**model_config)
model_cnn.compile()
model_cnn.summary()


# sample_shift({'sample_rate': 1/10, 'shift_max': 10})
