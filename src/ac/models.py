from keras.layers import Dense
"""
Model functions take:

input_placeholder, input placeholder to build model on
output_size, size of the output layer
name, the name to add to the layers
* Optional
n_layers, number of hidden layers
size, the size of hidden layers
activation, the activation to use in hidden layers

And return:

Model output tensor
"""


def fc(input_placeholder, output_size, name, n_layers=2, size=64, activation="relu"):
    x = input_placeholder
    for i in range(n_layers):
        x = Dense(size, activation=activation, name="{}-Dense{}".format(name, i))(x)
    x = Dense(output_size, name="{}-Output".format(name))(x)

    return x
