from matplotlib.pyplot import draw
from urllib3 import disable_warnings
from evaluation import *
import wandb
from wandb.keras import WandbCallback


def create_mlp_model():
    return tf.keras.models.Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(10, activation='softmax')
    ])


def create_cnn_model():
    return tf.keras.models.Sequential([
        Conv2D(20, 5, padding='same', activation='relu',
               input_shape=(28, 28, 1)),
        MaxPool2D(pool_size=(2, 2), padding='same'),
        Conv2D(50, 5, padding='same', activation='relu'),
        MaxPool2D(pool_size=(2, 2), padding='same'),
        Flatten(),
        Dense(500, activation='relu'),
        Dense(10, activation='softmax')
    ])


if __name__ == '__main__':

    config = {
        "model": {
            "hidden_layer_sizes": [16, 16],
            "activation": 'relu'
        },
        "train": {
            "model": 'MLP',
            "dataset": "mnist",
            "learning_rate": 1e-3,
            "epochs": 20,
            "batch_size": 32
        }}

    model_mlp = train_model(create_mlp, 'MLP-1', config)
    config = {
        'max_shift': 5,
        'extrapolation': False
    }
    evaluate_model(model_mlp, 'MLP-1', config)
