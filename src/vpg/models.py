from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten

"""
Model should be a function that takes an input placeholder, input_placeholder, and dimensions of output layer, 
output_size, and returns action logits of shape (None, output_size) of a model built on the input placeholder.

Returns outputs of a fully connected linear (no activation) layer which can be used
for example to parameterise a Gaussian or a softmax for continuous and discrete actions.

We use tanh activation on fully connected layers.
"""

def cnn_medium(input_placeholder, output_size):
    x = Conv2D(32, (3, 3), activation="relu")(input_placeholder)
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(output_size)(x)

    return x


def cnn_small(input_placeholder, output_size):
    x = Conv2D(32, (3, 3), activation="relu")(input_placeholder)
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = Flatten()(x)
    x = Dense(64, activation="tanh")(x)
    x = Dense(output_size)(x)

    return x


def fc_small(input_placeholder, output_size):
    x = Dense(64, activation="tanh")(input_placeholder)
    x = Dense(64, activation="tanh")(x)
    x = Dense(output_size)(x)

    return x


def fc_medium(input_placeholder, output_size):
    x = Dense(32, activation="tanh")(input_placeholder)
    x = Dense(64, activation="tanh")(x)
    x = Dense(32, activation="tanh")(x)
    x = Dense(output_size)(x)

    return x