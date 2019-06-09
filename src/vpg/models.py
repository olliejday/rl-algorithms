import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda

"""
Model should be a function that takes an input placeholder, input_placeholder, and dimensions of output layer, 
output_size, and returns action logits of shape (None, output_size) of a model built on the input placeholder.

Returns outputs of a fully connected linear (no activation) layer which can be used
for example to parameterise a Gaussian or a softmax for continuous and discrete actions.

We use tanh activation on fully connected layers.
"""


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


#############

from src.vpg.utils import gather_nd, gaussian_log_likelihood


class DiscretePolicy(tf.keras.Model):
    def __init__(self, hidden_layer_sizes, output_size, activation="relu", **kwargs):
        super(DiscretePolicy, self).__init__(**kwargs)
        self.model = tf.keras.Sequential()
        for hidden_units in hidden_layer_sizes:
            self.model.add(Dense(hidden_units, activation=activation))
        self.model.add(Dense(output_size, activation=None))

    def call(self, inputs):
        x = self.model(inputs)
        sampled_ac = Lambda(lambda x: tf.squeeze(tf.random.categorical(x, 1, name="sampled_ac", dtype=tf.int32),
                                     axis=1), name="sample_ac")(x)
        return sampled_ac

    def logprob(self, inputs, acs, name="logprob_ac"):
        # we apply a softmax to get the log probabilities in discrete case
        x = self.model(inputs)
        logprob = Lambda(lambda x: tf.nn.log_softmax(x), name="logprob")(x)
        logprob_acs = Lambda(lambda x: gather_nd(x, acs, name=name), name=name)(logprob)
        return logprob_acs


class ContinuousPolicy(tf.keras.Model):
    def __init__(self, hidden_layer_sizes, output_size, activation="relu", **kwargs):
        super(ContinuousPolicy, self).__init__(**kwargs)
        self.sy_logstd = tf.get_variable(name="log_std", shape=[output_size])
        self.model = tf.keras.Sequential()
        for hidden_units in hidden_layer_sizes:
            self.model.add(Dense(hidden_units, activation=activation))
        self.model.add(Dense(output_size, activation=None))

    def call(self, inputs):
        x = self.model(inputs)
        sample_z = tf.random.normal(shape=tf.shape(x))
        sampled_ac = Lambda(lambda x: x + tf.exp(self.sy_logstd) * sample_z, name="sample_ac")(x)
        return sampled_ac

    def logprob(self, inputs, acs, name="logprob"):
        x = self.model(inputs)
        logprob_acs = Lambda(lambda x: gaussian_log_likelihood(acs, x, self.sy_logstd), name=name)(x)
        return logprob_acs