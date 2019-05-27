from keras.layers import Conv2D, Dense, Flatten

"""
Note the format of these models is to use Keras as weight sharing.
So we __init__ with the model structure, and it can then be called repeatedly on different Keras Input() or 
TF Placeholders.

Example usage, where ModelClass is one of the models here.

from Keras import Input, Model

# Init class
model_fn = ModelClass(self.ac_dim)
# Use with inputs
input_placeholder = Input(shape=[x, y, z])
model_outputs = model_fn(self.ob_placeholder_float)
model = Model(self.ob_placeholder, self.q_func_ob)
"""

class DQNCNNModelKerasSmall:
    """
    Architecture inspired by original DeepMind paper, adjusted the strides and kernels to suit smaller grid sizes.
    """
    def __init__(self, output_size):
        self.model_in = Conv2D(32, kernel_size=3, strides=(1, 1), activation="relu")
        self.model_hidden = [Conv2D(64, kernel_size=2, strides=(2, 2), activation="relu"),
                             Conv2D(64, kernel_size=2, strides=(1, 1), activation="relu"),
                             Flatten(), Dense(264, activation="relu")]
        self.model_out = Dense(output_size)

    def __call__(self, ob_placeholder):
        x = self.model_in(ob_placeholder)
        for layer in self.model_hidden:
            x = layer(x)
        return self.model_out(x)

class DQNFCModelKeras:
    """
    A simple fully connected model.
    """
    def __init__(self, output_size):
        self.model_in = Dense(64, activation="relu")
        self.model_hidden = [Dense(64, activation="relu")]
        self.model_out = Dense(output_size)

    def __call__(self, ob_placeholder):
        x = self.model_in(ob_placeholder)
        for layer in self.model_hidden:
            x = layer(x)
        return self.model_out(x)
