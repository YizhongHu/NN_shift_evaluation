from evaluation import *

config = {
    "model": {
        "hidden_layer_sizes": [16, 16],
        "activation": 'relu'
    },
    "train": {
        "model": 'NLP',
        "dataset": "mnist",
        "learning_rate": 1e-3,
        "epochs": 20,
        "batch_size": 32
    }}

train_model(create_mlp, config)