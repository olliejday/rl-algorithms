import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda

from src.vpg.utils import gather_nd, gaussian_log_likelihood


"""
Models here should subclass the tf.keras.Models class.

We use tanh activation on fully connected layers.
"""


class FC_NN(tf.keras.Model):
    def __init__(self, hidden_layer_sizes, output_size, activation="tanh", **kwargs):
        super(FC_NN, self).__init__(**kwargs)
        self.model = tf.keras.Sequential()
        for hidden_units in hidden_layer_sizes:
            self.model.add(Dense(hidden_units, activation=activation))
        self.model.add(Dense(output_size, activation=None))
        self.model.add(Lambda(lambda x: tf.squeeze(x)))

    def call(self, inputs):
        return self.model(inputs)



class DiscretePolicy(tf.keras.Model):
    def __init__(self, hidden_layer_sizes, output_size, activation="tanh", **kwargs):
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
    def __init__(self, hidden_layer_sizes, output_size, activation="tanh", **kwargs):
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