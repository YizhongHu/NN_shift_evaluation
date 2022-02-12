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
    model_mlp = create_mlp_model()
    init_model(model_mlp)
    train(model_mlp, x_train, y_train, x_test, y_test,
          'mlp', epoch=20, model_path='./models')
    # model_mlp.load_weights('/content/drive/MyDrive/models/mnist/mlp/20220210-213904.ckpt')
    model_mlp.evaluate(x_test, y_test, callbacks=[WandbCallback()])
    accuracies_mlp = accuracy_on_shift(model_mlp)
    wandb.define_metric('MSE_5', summary='max')
    wandb.run.summary['accuracies'] = accuracies_mlp
    wandb.run.summary['MSE_5'] = mean_squared_error(accuracies_mlp)
    save_path = draw_accuracy(accuracies_mlp, 'MLP')
    with open(save_path) as html:
        wandb.log({'accuracies_on_shift': wandb.Html(html)})