import os

import tensorflow as tf
import time

from rl_algorithms.src.sac.models import GaussianPolicy, ValueFunction, QFunction


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
                 two_qf=True,
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
        self._two_qf = two_qf
        self._tau = tau

        self._training_ops = []

        self.experiments_dir = experiments_dir

        # make directory to save models
        if self.experiments_dir != "":
            save_dir = os.path.join(self.experiments_dir, "models")
            if not os.path.exists(save_dir):
                print("Made model directory: {}".format(save_dir))
                os.makedirs(save_dir)

    def build(self, env, q_function_params, value_function_params, policy_params):

        self.q_function = QFunction(name='q_function', **q_function_params)
        if self._two_qf:
            self.q_function2 = QFunction(name='q_function2', **q_function_params)
        else:
            self.q_function2 = None
        self.value_function = ValueFunction(
            name='value_function', **value_function_params)
        self.target_value_function = ValueFunction(
            name='target_value_function', **value_function_params)
        self.policy = GaussianPolicy(
            action_dim=env.action_space.shape[0],
            reparameterize=self._reparameterize,
            **policy_params)

        self._create_placeholders(env)

        policy_loss = self._policy_loss_for()
        value_function_loss = self._value_function_loss_for()
        q_function_loss = self._q_function_loss_for(self.q_function)
        if self.q_function2 is not None:
            q_function2_loss = self._q_function_loss_for(self.q_function2)

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


    def _q_function_loss_for(self, q_function):
        q_value_estimates = q_function([self._observations_ph, self._actions_ph])
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
                self.sess.run(self._training_ops, feed_dict)
                self.sess.run(self._target_update_ops)

            yield epoch

    def save_model(self, timestep):
        """
        Save current policy model.
        """
        if self.experiments_dir != "":
            fpath = os.path.join(self.experiments_dir, "models", "model-{}.h5".format(timestep))
            self.policy.save_weights(fpath)
        else:
            print("No experiments_dir, so cannot save model.")

    def load_model(self, model_path=None):
        """
        Load a model.
        If no path passed, loads the latest model.
        """
        if model_path is None:
            # then get latest model
            models_dir = os.path.join(self.experiments_dir, "models")
            model_files = os.listdir(models_dir)
            model_number = max([int(f.split(".")[0].split("-")[1]) for f in model_files])
            model_path = os.path.join(models_dir, "model-{}.h5".format(model_number))
        self.policy.load_weights(model_path)


def run_model(env, experiments_dir, model_path=None, n_episodes=3, **kwargs):
    """
    Run a saved, trained model.
    :param env: environment to run in
    :param experiments_path: the path to the experiments directory, with logs and models
    :param model_path: file path of model to run, if None then latest model in experiments_path is loaded
    :param n_episodes: number of episodes to run
    :param **kwargs: for VPG setup
    """

    sac = SAC(experiments_dir=experiments_dir, **kwargs)

    value_function_params = {
        'hidden_layer_sizes': (128, 128),
    }

    q_function_params = {
        'hidden_layer_sizes': (128, 128),
    }

    policy_params = {
        'hidden_layer_sizes': (128, 128),
    }

    sac.build(
        env=env,
        q_function_params=q_function_params,
        value_function_params=value_function_params,
        policy_params=policy_params)

    sac.load_model(model_path)

    for i in range(n_episodes):
        done = False
        obs = env.reset()
        env.render()
        reward = 0
        while not done:
            ac = sac.policy.eval(obs)
            obs, rew, done, _ = env.step(ac)
            reward += rew
            env.render()
            time.sleep(0.01)
        print("Reward: {}".format(reward))