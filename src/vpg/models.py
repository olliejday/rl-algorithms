from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten

"""
Model should be a function that takes an input place holder and an output size and returns the model outputs
for a model built on the input placeholder.
"""

def cnn_medium(input_placeholder, output_size):
    """
    Discrete action policy model.
    Builds a CNN model ontop of an input placeholder.
    Returns outputs of a fully connected linear (no activation) layer which can be used
    for example to parameterise a Gaussian or a softmax for continuous and discrete actions.
    :param input_placeholder: placeholder for inputs
    :param output_size: dimension of output layer
    :return: action logits (None, output_size)
    """
    x = Conv2D(32, (3, 3), activation="relu")(input_placeholder)
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(output_size, kernel_initializer='zeros')(x)

    return x


def cnn_small(input_placeholder, output_size):
    """
    Discrete action policy model.
    Builds a CNN model ontop of an input placeholder.
    Returns outputs of a fully connected linear (no activation) layer which can be used
    for example to parameterise a Gaussian or a softmax for continuous and discrete actions.
    :param input_placeholder: placeholder for inputs
    :param output_size: dimension of output layer
    :return: action logits (None, output_size)
    """
    x = Conv2D(32, (3, 3), activation="relu")(input_placeholder)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(output_size, kernel_initializer='zeros')(x)

    return x


def fc_small(input_placeholder, output_size):
    """
    Discrete action policy model.
    Builds a fully connected model ontop of an input placeholder.
    Returns outputs of a fully connected linear (no activation) layer which can be used
    for example to parameterise a Gaussian or a softmax for continuous and discrete actions.
    :param input_placeholder: placeholder for inputs
    :param output_size: dimension of output layer
    :return: action logits (None, output_size)
    """
    x = Dense(64, activation="relu")(input_placeholder)
    x = Dense(64, activation="relu")(x)
    x = Dense(output_size, kernel_initializer='zeros')(x)

    return x
