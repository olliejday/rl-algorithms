import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Conv2D

from rl_algorithms.src.common.utils import gather_nd, gaussian_log_likelihood



def build_fc(hidden_layer_sizes, output_size, activation="relu", output_activation=None, squeeze=False):
    model = tf.keras.Sequential()
    for hidden_units in hidden_layer_sizes:
        model.add(Dense(hidden_units, activation=activation))
    model.add(Dense(output_size, activation=output_activation))
    if squeeze:
        model.add(Lambda(lambda x: tf.squeeze(x)))
    return model


def build_cnn(hidden_layer_sizes, filter_shapes, strides, output_size, activation="relu",
              output_activation=None, flatten=True, squeeze=True):
    model = tf.keras.Sequential()
    for units, size, stride in zip(hidden_layer_sizes, filter_shapes, strides):
        model.add(Conv2D(units, size, strides=stride, activation=activation))
    if flatten:
        model.add(Dense(output_size, activation=output_activation))
        if squeeze:
            model.add(Lambda(lambda x: tf.squeeze(x)))
    return model


class FC_NN(tf.keras.Model):
    def __init__(self, hidden_layer_sizes, output_size, activation="relu", output_activation=None,
                 squeeze=True, **kwargs):
        """
        :param hidden_layer_sizes: list of ints for hidden layer number of units
        :param output_size: number of nodes in output layer
        :param activation: activation for hidden nodes (default=relu)
        :param output_activation: activation for output layer (default=None)
        :param squeeze: whether to squeeze the output layer, best to for value fns
        :param kwargs: other arguments for Keras
        """
        super(FC_NN, self).__init__(**kwargs)
        self.model = build_fc(hidden_layer_sizes, output_size, activation, output_activation, squeeze)

    def call(self, inputs):
        return self.model(inputs)


class CNN(tf.keras.Model):
    def __init__(self, hidden_layer_sizes, filter_shapes, strides, output_size, activation="relu",
                 output_activation=None, flatten=True, squeeze=True, **kwargs):
        """
        Let N be the number of hidden layers
        :param hidden_layer_sizes: list of N ints for hidden layer number of units
        :param filter_shapes: list of N ints for filter sizes
        :param strides: list of N ints for strides
        :param output_size: number of nodes in output layer
        :param activation: activation for hidden nodes (default=relu)
        :param output_activation: activation for output layer (default=None)
        :param flatten: whether to flatten the conv layer and pass it through a dense layer with output_size
        units
        :param squeeze: whether to squeeze the output layer, best to for value fns
        :param kwargs: other arguments for Keras
        """
        super(CNN, self).__init__(**kwargs)
        self.model = build_cnn(hidden_layer_sizes, filter_shapes, strides, output_size, activation,
                               output_activation, flatten, squeeze)

    def call(self, inputs):
        return self.model(inputs)

# TODO: need to change policy to CNN (use above fn) might then be worth testing VIGAN + CNN PPO on cartpole and
#   or lander since before was CNN discrim, but FC PPO

class DiscretePolicyFC(tf.keras.Model):
    def __init__(self, hidden_layer_sizes, output_size, activation="relu", **kwargs):
        super(DiscretePolicyFC, self).__init__(**kwargs)
        self.model = build_fc(hidden_layer_sizes, output_size, activation)

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


class ContinuousPolicyFC(tf.keras.Model):
    def __init__(self, hidden_layer_sizes, output_size, activation="relu", **kwargs):
        super(ContinuousPolicyFC, self).__init__(**kwargs)
        self.sy_logstd = tf.Variable(name="log_std", initial_value=tf.zeros(output_size))
        self.model = self.model = build_fc(hidden_layer_sizes, output_size, activation)

    def call(self, inputs):
        x = self.model(inputs)
        sample_z = tf.random.normal(shape=tf.shape(x))
        sampled_ac = Lambda(lambda x: x + tf.exp(self.sy_logstd) * sample_z, name="sample_ac")(x)
        return sampled_ac

    def logprob(self, inputs, acs, name="logprob"):
        x = self.model(inputs)
        logprob_acs = Lambda(lambda x: gaussian_log_likelihood(acs, x, self.sy_logstd), name=name)(x)
        return logprob_acs


class DiscretePolicyCNN(tf.keras.Model):
    def __init__(self, hidden_layer_sizes, output_size, activation="relu", **kwargs):
        super(DiscretePolicyCNN, self).__init__(**kwargs)
        self.model = build_cnn(hidden_layer_sizes, output_size, activation)

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


class ContinuousPolicyCNN(tf.keras.Model):
    def __init__(self, hidden_layer_sizes, output_size, activation="relu", **kwargs):
        super(ContinuousPolicyCNN, self).__init__(**kwargs)
        self.sy_logstd = tf.Variable(name="log_std", initial_value=tf.zeros(output_size))
        self.model = self.model = build_cnn(hidden_layer_sizes, output_size, activation)

    def call(self, inputs):
        x = self.model(inputs)
        sample_z = tf.random.normal(shape=tf.shape(x))
        sampled_ac = Lambda(lambda x: x + tf.exp(self.sy_logstd) * sample_z, name="sample_ac")(x)
        return sampled_ac

    def logprob(self, inputs, acs, name="logprob"):
        x = self.model(inputs)
        logprob_acs = Lambda(lambda x: gaussian_log_likelihood(acs, x, self.sy_logstd), name=name)(x)
        return logprob_acs