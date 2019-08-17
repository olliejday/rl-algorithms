import tensorflow as tf
from tensorflow.keras.layers import Lambda
from rl_algorithms.src.common.utils import gaussian_log_likelihood, gather_nd
from rl_algorithms.src.common.models import build_cnn, build_fc


class FC_NN(tf.keras.Model):
    def __init__(self, dense_params, output_size, output_activation=None, squeeze=True, **kwargs):
        """
        :param dense_params: list of dicts where each dict is the arguments to tf.keras.layers.Dense to setup that layer
            see utils/models.py for details
        :param output_size: size of final output
        :param output_activation: activation for output layer
        :param squeeze: whether to squeeze the final output layer
        :param kwargs: other arguments for Keras
        """
        super(FC_NN, self).__init__(**kwargs)
        self.model = build_fc(dense_params, output_size, output_activation=output_activation, squeeze=squeeze)

    def call(self, inputs):
        return self.model(inputs)


class CNN(tf.keras.Model):
    def __init__(self, conv_params, dense_params, output_size, output_activation=None, squeeze=True, **kwargs):
        """
        :param cnn_params: list of dicts where each dict is the arguments to tf.keras.layers.Conv2D to setup that layer
            see utils/models.py for details
        :param dense_params: list of dicts where each dict is the arguments to tf.keras.layers.Dense to setup that layer
            if empty then no dense layers will be added to the conv layers
            see utils/models.py for details
        :param output_size: size of final output
        :param model: tf.keras.Sequential() model to add layers to or None for new model
        :param output_activation: activation for output layer
        :param squeeze: whether to squeeze the final output layer
        :return: model tf.keras.Sequential()
        """
        super(CNN, self).__init__(**kwargs)
        self.model = build_cnn(conv_params, dense_params, output_size, output_activation=output_activation,
                               squeeze=squeeze)

    def call(self, inputs):
        return self.model(inputs)


class DiscretePolicyFC(tf.keras.Model):
    def __init__(self, dense_params, output_size, **kwargs):
        super(DiscretePolicyFC, self).__init__(**kwargs)
        self.model = build_fc(dense_params, output_size)

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
    def __init__(self, dense_params, output_size, **kwargs):
        super(ContinuousPolicyFC, self).__init__(**kwargs)
        self.sy_logstd = tf.Variable(name="log_std", initial_value=tf.zeros(output_size))
        self.model = build_fc(dense_params, output_size)

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
    def __init__(self, conv_params, dense_params, output_size, **kwargs):
        super(DiscretePolicyCNN, self).__init__(**kwargs)
        self.model = build_cnn(conv_params, dense_params, output_size)

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
    def __init__(self, conv_params, dense_params, output_size, **kwargs):
        super(ContinuousPolicyCNN, self).__init__(**kwargs)
        self.sy_logstd = tf.Variable(name="log_std", initial_value=tf.zeros(output_size))
        self.model = build_cnn(conv_params, dense_params, output_size)

    def call(self, inputs):
        x = self.model(inputs)
        sample_z = tf.random.normal(shape=tf.shape(x))
        sampled_ac = Lambda(lambda x: x + tf.exp(self.sy_logstd) * sample_z, name="sample_ac")(x)
        return sampled_ac

    def logprob(self, inputs, acs, name="logprob"):
        x = self.model(inputs)
        logprob_acs = Lambda(lambda x: gaussian_log_likelihood(acs, x, self.sy_logstd), name=name)(x)
        return logprob_acs