import os

import tensorflow as tf
import time

from src.sac.models import GaussianPolicy, ValueFunction, QFunction


class SAC:
    """Soft Actor-Critic (SAC)
    Original code from Tuomas Haarnoja, Soroush Nasiriany, and Aurick Zhou for CS294-112 Fall 2018

    References
    ----------
    [1] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine, "Soft
        Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
        with a Stochastic Actor," ICML 2018.
    """

    def __init__(self,
                 experiments_dir="",
                 alpha=1.0,
                 batch_size=256,
                 discount=0.99,
                 epoch_length=1000,
                 learning_rate=3e-3,
                 reparameterize=False,
                 tau=0.01):
        """
        Args:
        """

        self._alpha = alpha
        self._batch_size = batch_size
        self._discount = discount
        self._epoch_length = epoch_length
        self._learning_rate = learning_rate
        self._reparameterize = reparameterize
        self._tau = tau

        self.experiments_dir = experiments_dir

        self._training_ops = []

    def build(self, env, two_qf, reparam, q_function_params, value_function_params, policy_params):

        self.q_function = QFunction(name='q_function', **q_function_params)
        if two_qf:
            self.q_function2 = QFunction(name='q_function2', **q_function_params)
        else:
            self.q_function2 = None
        self.value_function = ValueFunction(
            name='value_function', **value_function_params)
        self.target_value_function = ValueFunction(
            name='target_value_function', **value_function_params)
        self.policy = GaussianPolicy(
            action_dim=env.action_space.shape[0],
            reparameterize=reparam,
            **policy_params)

        self._create_placeholders(env)

        policy_loss = self._policy_loss_for()
        value_function_loss = self._value_function_loss_for()
        q_function_loss = self._q_function_loss_for()
        if self.q_function2 is not None:
            q_function2_loss = self._q_function_loss_for()

        optimizer = tf.train.AdamOptimizer(
            self._learning_rate, name='optimizer')
        policy_training_op = optimizer.minimize(
            loss=policy_loss, var_list=self.policy.trainable_variables)
        value_training_op = optimizer.minimize(
            loss=value_function_loss,
            var_list=self.value_function.trainable_variables)
        q_function_training_op = optimizer.minimize(
            loss=q_function_loss, var_list=self.q_function.trainable_variables)
        if self.q_function2 is not None:
            q_function2_training_op = optimizer.minimize(
                loss=q_function2_loss, var_list=self.q_function2.trainable_variables)

        self._training_ops = [
            policy_training_op, value_training_op, q_function_training_op
        ]
        if self.q_function2 is not None:
            self._training_ops += [q_function2_training_op]
        self._target_update_ops = self._create_target_update(
            source=self.value_function, target=self.target_value_function)

        self.init_tf()

    def init_tf(self):
        self.sess = tf.keras.backend.get_session()
        self.sess.run(tf.global_variables_initializer())

    def _create_placeholders(self, env):
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, observation_dim),
            name='observation',
        )
        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, observation_dim),
            name='next_observation',
        )
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, action_dim),
            name='actions',
        )
        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='rewards',
        )
        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='terminals',
        )

    def _policy_loss_for(self):
        if self._reparameterize:
            # normal sample stage handled within policy
            action_samples, sample_log_probs = self.policy(self._observations_ph)
            q_value_estimates = self.q_function([self._observations_ph, action_samples])
            return tf.reduce_mean(self._alpha * sample_log_probs - q_value_estimates)
        else:
            action_samples, sample_log_probs = self.policy(self._observations_ph)
            q_value_estimates = self.q_function([self._observations_ph, action_samples])
            if self.q_function2 is not None:
                # correct for positive bias with two q functions
                q_value_estimates = tf.minimum(q_value_estimates,
                                               self.q_function2([self._observations_ph, action_samples]))
            # baseline as value function
            baseline = self.value_function(self._observations_ph)
            # don't want value through targets
            target_values = tf.stop_gradient(self._alpha * sample_log_probs - q_value_estimates + baseline)
            # we do want gradients through this policy sample
            return tf.reduce_mean(sample_log_probs * target_values)

    def _value_function_loss_for(self):
        action_samples, sample_log_probs = self.policy(self._observations_ph)
        value_function_estimates = self.value_function(self._observations_ph)
        # correct for positive bias with two q functions
        q_function_estimates = self.q_function([self._observations_ph, action_samples])
        if self.q_function2 is not None:
            # correct for positive bias with two q functions
            q_function_estimates = tf.minimum(q_function_estimates,
                                              self.q_function2([self._observations_ph, action_samples]))
        return tf.reduce_mean((value_function_estimates - (q_function_estimates - self._alpha * sample_log_probs)) ** 2)


    def _q_function_loss_for(self):
        q_value_estimates = self.q_function([self._observations_ph, self._actions_ph])
        target_value_estimates = self.target_value_function(self._next_observations_ph)
        # incorporate discount and the terminal mask
        target_values = self._rewards_ph + (1 - self._terminals_ph) * self._discount * target_value_estimates
        return tf.reduce_mean((q_value_estimates - target_values) ** 2)

    def _create_target_update(self, source, target):
        """Create tensorflow operations for updating target value function."""

        return [
            tf.assign(target, (1 - self._tau) * target + self._tau * source)
            for target, source in zip(target.trainable_variables, source.
                                      trainable_variables)
        ]

    def train(self, sampler, n_epochs=1000):
        """Performs RL training using the `sampler` for `n_epochs`.
        """
        self._start = time.time()
        for epoch in range(n_epochs):
            for t in range(self._epoch_length):
                sampler.sample()

                batch = sampler.random_batch(self._batch_size)
                feed_dict = {
                    self._observations_ph: batch['observations'],
                    self._actions_ph: batch['actions'],
                    self._next_observations_ph: batch['next_observations'],
                    self._rewards_ph: batch['rewards'],
                    self._terminals_ph: batch['terminals'],
                }
                tf.get_default_session().run(self._training_ops, feed_dict)
                tf.get_default_session().run(self._target_update_ops)

            yield epoch

    def save_model(self, timestep):
        """
        Save current policy model.
        """
        if self.experiments_dir != "":
            fpath = os.path.join(self.experiments_dir, "models", "model-{}.h5".format(timestep))
            self.policy.save_weights(fpath)
