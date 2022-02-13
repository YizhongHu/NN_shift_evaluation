from evaluation import *

exp_name = 'CNN-shift-2'
dataset_name = 'mnist-shift'
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
config = {
    "model": model_config,
    "train": {
        "model": 'CNN',
        "dataset": dataset_name,
        "learning_rate": 1e-3,
        "epochs": 5,
        "batch_size": 32
    }}

model_cnn = train_model(create_cnn, exp_name, config)
config = {
    'max_shift': 10,
    'extrapolation': True,
    'train_shift': 5,
    'dataset': dataset_name
}
evaluate_model(model_cnn, exp_name, config)
