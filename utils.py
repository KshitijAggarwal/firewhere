import pylab as plt
import tensorflow as tf


def plot_loss(history):
    """

    Args:
        history:

    Returns:

    """
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error [MPG]")
    plt.legend()
    plt.grid(True)


def make_model(trial, normalizer):
    """

    Args:
        trial:
        normalizer:

    Returns:

    """
    sizes = [32, 64, 128, 256, 512, 1024]
    n_layers = trial.suggest_int("n_layers", 4, 8)
    units = [normalizer]
    for i in range(n_layers):
        n_units = trial.suggest_categorical(f"n_nodes_l{i}", sizes)
        units.append(tf.keras.layers.Dense(n_units, activation="relu"))
    units.append(tf.keras.layers.Dense(1))
    return tf.keras.Sequential(units)
