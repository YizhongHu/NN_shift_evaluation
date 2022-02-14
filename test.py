from evaluation import *

exp_name = 'CNN-shift-2'
dataset_name = 'mnist'
model_config = {
    'conv_size': (3, 3),
    'conv_layers': [64, 128, 256],
    'conv_padding': 'same',
    'pool_size': (2, 2),
    'pool_type': 'max',
    'global_pool': 'average',
    'conv_dropout': 0.0,
    'hidden_layers': [500],
    'dense_dropout': 0.0,
    'activation': 'relu'
}

model_cnn = create_cnn(**model_config)
model_cnn.compile()
model_cnn.summary()
