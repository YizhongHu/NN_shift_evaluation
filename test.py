from evaluation import *
from dataset import *
exp_name = 'LL-opt6-7x7-1Dense'
dataset_name = 'mnist-pad'
model_config = {
    'input_shape': (48, 48, 1),
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
    'activation': 'relu'
}
model_cnn = create_cnn(**model_config)
model_cnn.compile()
model_cnn.summary()

# sample_shift({'sample_rate': 1/10, 'shift_max': 10})


# config = {
#     'duplicate': 1,
#     'roll_max': 10
# }
# roll_loc_data(config)