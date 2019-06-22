import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda

"""
Setup the model

Init takes:
:param output_size: size of output layer (number of actions) 
:param integer_observations: if using integer observations we add a cast to float layer
:param env: environment to get observation space from. If using integer observations must be passed. 
"""


class DQNCNNModelKerasSmall(tf.keras.Model):
    """
    Architecture inspired by original DeepMind paper, adjusted the strides and kernels to suit smaller grid sizes.
    """

    def __init__(self, output_size, integer_observations=False, env=None, **kwargs):
        super(DQNCNNModelKerasSmall, self).__init__(**kwargs)
        self.model = tf.keras.Sequential()
        if integer_observations:
            assert env is not None, "If integer observations, must pass env to model.__init__()"
            ob_range = env.observation_space.high - env.observation_space.low
            self.model.add(Lambda(lambda x: tf.cast(x, tf.float32) / ob_range))
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

    def __init__(self, output_size, integer_observations=False, env=None, **kwargs):
        super(DQNFCModelKeras, self).__init__(**kwargs)
        self.model = tf.keras.Sequential()
        if integer_observations:
            assert env is not None, "If integer observations, must pass env to model.__init__()"
            ob_range = env.observation_space.high - env.observation_space.low
            self.model.add(Lambda(lambda x: tf.cast(x, tf.float32) / ob_range))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(output_size))

    def call(self, inputs):
        return self.model(inputs)
