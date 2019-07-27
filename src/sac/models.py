import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions


### GENERAL


class ValueFunction(tf.keras.Model):
    def __init__(self, hidden_layer_sizes, **kwargs):
        super(ValueFunction, self).__init__(**kwargs)
        self.model = tf.keras.Sequential()
        for hidden_units in hidden_layer_sizes:
            self.model.add(tf.keras.layers.Dense(hidden_units, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation=None))
        self.model.add(tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1)))

    def call(self, inputs):
        return self.model(inputs)


### CONTINUOUS


class QFunctionContinuous(tf.keras.Model):
    def __init__(self, hidden_layer_sizes, **kwargs):
        super(QFunctionContinuous, self).__init__(**kwargs)
        self.model = tf.keras.Sequential()
        for hidden_units in hidden_layer_sizes:
            self.model.add(tf.keras.layers.Dense(hidden_units, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation=None))
        self.model.add(tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1)))

    def call(self, inputs):
        x = tf.keras.layers.Concatenate(axis=1)(list(inputs))
        return self.model(x)


class GaussianPolicy(tf.keras.Model):
    def __init__(self, action_dim, hidden_layer_sizes, reparameterize, **kwargs):
        super(GaussianPolicy, self).__init__(**kwargs)
        self._f = None
        self._reparameterize = reparameterize
        self.model = tf.keras.Sequential()
        self.action_dim = action_dim
        for hidden_units in hidden_layer_sizes:
            self.model.add(tf.keras.layers.Dense(hidden_units, activation='relu'))
        self.model.add(tf.keras.layers.Dense(action_dim * 2, activation=None))

    def call(self, inputs):
        mean_and_log_std = self.model(inputs)


        mean, log_std = tf.split(
            mean_and_log_std, num_or_size_splits=2, axis=1)
        log_std = tf.clip_by_value(log_std, -20., 2.)

        distribution = distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(log_std))

        raw_actions = distribution.sample()
        if not self._reparameterize:
            # if not reparameterising we treat sampled actions as constants
            raw_actions = tf.stop_gradient(raw_actions)
        log_probs = distribution.log_prob(raw_actions)
        log_probs -= self._squash_correction(raw_actions)

        actions = tf.tanh(raw_actions)

        return [actions, log_probs]

    def _squash_correction(self, raw_actions, stable=True, eps=1e-5):
        # change of variables correction for log likelihood - log det jac (see assignment)
        # with small constant in the log term for numerical stability
        if not stable:
            return tf.reduce_sum(tf.log(1 - tf.tanh(raw_actions) ** 2 + eps), axis=1)
        # the numerically stable version
        return tf.reduce_sum(np.log(4) - 2 * tf.nn.softplus(2 * raw_actions) + 2 * raw_actions, axis=1)

    def eval(self, observation):
        assert self.built and observation.ndim == 1

        if self._f is None:
            self._f = tf.keras.backend.function(self.inputs, [self.outputs[0]])

        action, = self._f([observation[None]])
        return action.flatten()


### DISCRETE


# TODO: discrete models