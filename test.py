from evaluation import *

exp_name = 'CNN-shift-2'
dataset_name = 'mnist'
model_config = {
    "hidden_layer_sizes": [800],
    "activation": 'relu'
}

model_cnn = create_mlp(**model_config)
model_cnn.compile()
model_cnn.summary()
