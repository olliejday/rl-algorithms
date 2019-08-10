import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions

from src.common.utils import gather_nd


# TODO: from paper, reward scale is important
# TODO: from paper (and see baselines), evaluation with deterministic policy taking the mean can perform better

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

        # TODO: need to rescale actions to match env.
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


class QFunctionDiscrete(tf.keras.Model):
    """
    DQN style Q function, where the state is input and values for all actions are computed.

    ie. Q(s) -> [a1, a2, ..., aN]
    """

    def __init__(self, ac_dim, hidden_layer_sizes, **kwargs):
        super(QFunctionDiscrete, self).__init__(**kwargs)
        self.model = tf.keras.Sequential()
        for hidden_units in hidden_layer_sizes:
            self.model.add(tf.keras.layers.Dense(hidden_units, activation='relu'))
        self.model.add(tf.keras.layers.Dense(ac_dim, activation=None))

    def call(self, inputs):
        """
        Sets up ops for Q value of state, action for inputs [state, action]
        Returns Q function estimate.
        """
        state, action = inputs
        q_values = self.q_values(state)
        q_value_ac = tf.keras.layers.Lambda(lambda x: gather_nd(x, action, name="q_value_ac"),
                                            name="q_value_ac")(q_values)
        return q_value_ac

    def q_values(self, state):
        """
        Sets up ops to get Q function estimates of all actions for input state `inputs`.
        Returns array of Q values (ac_dim,)
        """
        q_values = self.model(state)
        return q_values


class QFunctionDiscreteActionInputs(tf.keras.Model):
    """
    Q model where the state and action are concatenated as input and one q value is output.

    ie. Q(s, a) -> q val

    Action inputs are one hot encoded.

    NOTE: this doesn't work very well in our experiments but is included for completeness.
    """

    def __init__(self, ac_dim, hidden_layer_sizes, **kwargs):
        super(QFunctionDiscreteActionInputs, self).__init__(**kwargs)
        self.ac_dim = ac_dim
        self.model = tf.keras.Sequential()
        for hidden_units in hidden_layer_sizes:
            self.model.add(tf.keras.layers.Dense(hidden_units, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation=None))

    def call(self, inputs):
        """
        Sets up ops for Q value of state, action for inputs [state, action]
        Returns Q function estimates (batch size, 1).
        """
        state, action = inputs
        action_encoded = self.encode_action(action)
        model_inputs = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=1))([state, action_encoded])
        q_value = self.model(model_inputs)
        return q_value

    def q_values(self, state):
        """
        Sets up ops to get Q function estimates of all actions for input state `inputs`.
        Returns array of Q values (batch size, ac_dim)
        """
        q_values = []
        for ac in range(self.ac_dim):
            # get one hot action for `ac` stacked for N times for N states input
            b_size = tf.shape(state)[0]
            acs = tf.ones(b_size, dtype=tf.int32) * ac
            q_values.append(self.call([state, acs]))
        q_values_stacked = tf.stack(q_values, axis=1)
        return q_values_stacked

    def encode_action(self, acs):
        """
        Ops to one-hot encodes an action for input
        :param acs: actions tensor
        :return: one-hot encoded action
        """
        actions_encoded = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.one_hot(tf.keras.backend.cast(x, 'int64'), self.ac_dim))(acs)
        return actions_encoded


class CategoricalPolicy(tf.keras.Model):
    def __init__(self, action_dim, hidden_layer_sizes, **kwargs):
        super(CategoricalPolicy, self).__init__(**kwargs)
        self._f = None
        self.model = tf.keras.Sequential()
        self.action_dim = action_dim
        for hidden_units in hidden_layer_sizes:
            self.model.add(tf.keras.layers.Dense(hidden_units, activation='relu'))
        self.model.add(tf.keras.layers.Dense(action_dim, activation=None))

    def call(self, inputs):
        """
        Setup ops to call policy model on `inputs`
        Returns [sampled_actions, log_probs (of the samples acs)]
        """
        x = self.model(inputs)
        logprob = tf.keras.layers.Lambda(lambda x: tf.nn.log_softmax(x), name="logprob")(x)
        self.sampled_ac = tf.keras.layers.Lambda(
            lambda x: tf.squeeze(tf.random.categorical(x, 1, name="sampled_ac", dtype=tf.int32)), name="sample_ac")(x)
        logprob_sampled = tf.keras.layers.Lambda(lambda x: gather_nd(x, self.sampled_ac, name="logprob_sampled"),
                                                 name="logprob_sampled")(logprob)
        return self.sampled_ac, logprob_sampled

    def logprobs(self, inputs):
        """
        Sets up ops to get log prob of all actions for input state `inputs`.
        Returns array of logprobs (ac_dim,)
        """
        x = self.model(inputs)
        logprob = tf.keras.layers.Lambda(lambda x: tf.nn.log_softmax(x), name="logprob")(x)
        return logprob

    def eval(self, observation):
        """
        Given `observations`, return actions
        """
        assert self.built and observation.ndim == 1

        if self._f is None:
            self._f = tf.keras.backend.function(self.inputs, self.sampled_ac)

        action = self._f([observation[None]])
        return action
