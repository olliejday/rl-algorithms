import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Lambda

# TODO: adapt code to use these calls for building models

def build_fc(dense_params, output_size, model=None, output_activation=None, squeeze=False):
    """
    Build a Fully connected (dense) tf.keras.Sequential() model
    :param dense_params: list of dicts where each dict is the arguments to tf.keras.layers.Dense to setup that layer
        each layer dict must have "units" key for number of units in the layer
        recommended to also set an "activation" as default is None ie. linear
    :param output_size: size of final output
    :param model: tf.keras.Sequential() model to add layers to or None for new model
    :param output_activation: activation for output layer
    :param squeeze: whether to squeeze the final output layer
    :return: model tf.keras.Sequential()
    """
    if model is None:
        model = tf.keras.Sequential()
    for param in dense_params:
        model.add(Dense(**param))
    model.add(Dense(output_size, activation=output_activation))
    if squeeze:
        model.add(Lambda(lambda x: tf.squeeze(x)))
    return model


def build_cnn(cnn_params, dense_params, output_size, model=None, output_activation=None, squeeze=False):
    """
    Build a CNN tf.keras.Sequential() model
    :param cnn_params: list of dicts where each dict is the arguments to tf.keras.layers.Conv2D to setup that layer
        each layer dict must have "filters" and "kernel_size" keys set
        recommended to also set an "activation" as default is None ie. linear
    :param dense_params: list of dicts where each dict is the arguments to tf.keras.layers.Dense to setup that layer
        if empty then no dense layers will be added to the conv layers
        each layer dict must have "units" key for number of units in the layer
        recommended to also set an "activation" as default is None ie. linear
    :param output_size: size of final output
    :param model: tf.keras.Sequential() model to add layers to or None for new model
    :param output_activation: activation for output layer
    :param squeeze: whether to squeeze the final output layer
    :return: model tf.keras.Sequential()
    """
    if model is None:
        model = tf.keras.Sequential()
    for param in cnn_params:
        model.add(Conv2D(**param))
    if len(dense_params) > 0:
        model.add(Flatten())
        model = build_fc(dense_params, output_size, model=model, output_activation=output_activation, squeeze=squeeze)
    return model
