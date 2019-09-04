import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Lambda

# TODO: adapt code to use these calls for building models (see ppo.models)

def build_fc(dense_params, output_size, output_activation=None, squeeze=False):
    """
    Build a Fully connected (dense) model. Here the model is defined as a list of layers, which can then
    be used with the Sequential .add(layer) or the Functional API.
    :param dense_params: list of dicts where each dict is the arguments to tf.keras.layers.Dense to setup that layer
        each layer dict must have "units" key for number of units in the layer
        recommended to also set an "activation" as default is None ie. linear
    :param output_size: size of final output
    :param output_activation: activation for output layer
    :param squeeze: whether to squeeze the final output layer
    :return: list of keras layers.
    """
    model = []
    for param in dense_params:
        model.append(Dense(**param))
    if output_size is not None:
        model.append(Dense(output_size, activation=output_activation))
    if squeeze:
        model.append(Lambda(lambda x: tf.squeeze(x)))
    return model


def build_cnn(cnn_params, dense_params, output_size, output_activation=None, squeeze=False):
    """
    Build a CNN model. Here the model is defined as a list of layers, which can then
    be used with the Sequential .add(layer) or the Functional API.
    :param cnn_params: list of dicts where each dict is the arguments to tf.keras.layers.Conv2D to setup that layer
        each layer dict must have "filters" and "kernel_size" keys set
        recommended to also set an "activation" as default is None ie. linear
    :param dense_params: list of dicts where each dict is the arguments to tf.keras.layers.Dense to setup that layer
        if empty then no dense layers will be added to the conv layers
        each layer dict must have "units" key for number of units in the layer
        recommended to also set an "activation" as default is None ie. linear
    :param output_size: size of final output
    :param output_activation: activation for output layer
    :param squeeze: whether to squeeze the final output layer
    :return: list of keras layers.
    """
    model = []
    for param in cnn_params:
        model.append(Conv2D(**param))
    if len(dense_params) > 0:
        model.append(Flatten())
        model += build_fc(dense_params, output_size, output_activation=output_activation, squeeze=squeeze)
    return model
