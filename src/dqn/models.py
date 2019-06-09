import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten


class DQNCNNModelKerasSmall(tf.keras.Model):
    """
    Architecture inspired by original DeepMind paper, adjusted the strides and kernels to suit smaller grid sizes.
    """

    def __init__(self, output_size, **kwargs):
        super(DQNCNNModelKerasSmall, self).__init__(**kwargs)
        self.model = tf.keras.Sequential()
        self.model.add(Conv2D(32, kernel_size=3, strides=(1, 1), activation="relu"))
        self.model.add(Conv2D(64, kernel_size=2, strides=(2, 2), activation="relu"))
        self.model.add(Conv2D(64, kernel_size=2, strides=(1, 1), activation="relu"))
        self.model.add(Flatten())
        self.model.add(Dense(264, activation="relu"))
        self.model.add(Dense(output_size))

    def call(self, inputs):
        return self.model(inputs)


class DQNFCModelKeras(tf.keras.Model):
    """
    A simple fully connected model.
    """

    def __init__(self, output_size, **kwargs):
        super(DQNFCModelKeras, self).__init__(**kwargs)
        self.model = tf.keras.Sequential()
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(output_size))

    def call(self, inputs):
        return self.model(inputs)
