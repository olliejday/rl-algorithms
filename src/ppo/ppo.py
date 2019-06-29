import tensorflow as tf
import numpy as np
import time
import os
import gym

from src.ppo.utils import PPOBuffer
from src.ppo.models import DiscretePolicy, ContinuousPolicy


class ProximalPolicyOptimisation:
    def __init__(self,
                 env,
                 hidden_layer_sizes=[64, 64],
                 experiments_path="",
                 policy_learning_rate=5e-3,
                 value_fn_learning_rate=1e-2,
                 n_policy_updates=15,
                 n_value_updates=15,
                 clip_ratio=0.2,
                 value_fn_class=None,
                 render_every=20,
                 max_path_length=1000,
                 min_timesteps_per_batch=10000,
                 gamma=0.99,
                 gae_lambda=0.975,
                 normalise_advantages=True,
                 ):

        """
        Run Proximal Policy Optimisation (PPO) algorithm.

        Parameters
        ----------
        env: gym.Env
            gym environment to train on.
        hidden_layer_sizes: list
            List of ints for the number of units to have in the hidden layers
        experiments_path: string
            path to save models to during training
        policy_learning_rate: float
            Learning rate to train policy with.
        value_fn_learning_rate: float
            Learning rate to train with.
        n_policy_updates: int
            Number of policy updates to make every update iteration
        n_value_updates: int
            Number of value function updates to make every update iteration
        clip_ratio: float
            Hyperparameter for clipping policy objective
        value_fn_class: tf.keras.Model
            Model class to compute value function, see models.py.
            Should be __init__ and ready to call with inputs
        render_every: int
            Render an episode regularly through training to monitor progress
        max_path_length: int
            Max number of timesteps in an episode before stopping.
        min_timesteps_per_batch: int
            Min number of timesteps to gather for use in training updates.
        gamma: float
            Discount rate
        gae_lambda: float
            Lambda value for GAE (Between 0 and 1, close to 1)
        normalise_advantages: bool
            Whether to normalise advantages.
        """
        assert value_fn_class is not None, "Must provide value function."

        self.env = env
        # Is this env continuous, or self.discrete?
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.ob_dim = env.observation_space.shape
        if self.discrete:
            self.ac_dim = env.action_space.n
        else:
            self.ac_dim = env.action_space.shape[0]

        self.experiments_path = experiments_path

        self.hidden_layer_sizes = hidden_layer_sizes
        self.policy_learning_rate = policy_learning_rate
        self.value_fn_learning_rate = value_fn_learning_rate
        self.n_policy_updates = n_policy_updates
        self.n_value_fn_updates = n_value_updates
        self.clip_ratio = clip_ratio
        self.value_fn_class = value_fn_class
        self.render_every = render_every
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalise_advantages = normalise_advantages

        # make directory to save models
        if self.experiments_path != "":
            save_dir = os.path.join(self.experiments_path, "models")
            if not os.path.exists(save_dir):
                print("Made model directory: {}".format(save_dir))
                os.makedirs(save_dir)

    def __str__(self):
        """
        Define string behaviour as key parameters for logging
        """
        to_string = """
        policy_learning_rate: {}
        value_function_learning_rate: {}
        hidden_layer_size: {}
        nn_basline: {}
        max_path_length: {}
        min_timesteps_per_batch: {}
        gamma: {}
        normalise_advntages: {}""".format(
            self.policy_learning_rate,
            self.value_fn_learning_rate,
            self.hidden_layer_sizes,
            self.value_fn_class,
            self.max_path_length,
            self.min_timesteps_per_batch,
            self.gamma,
            self.normalise_advantages)
        return to_string

    def setup_placeholders(self):
        self.obs_ph = tf.placeholder(shape=[None] + [dim for dim in self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            self.acs_ph = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            self.acs_ph = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)

        self.adv_ph = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        # for PPO objective (also used for KL divergence estimate)
        self.prev_logprob_ph = tf.placeholder(shape=[None], name="prev_logprob", dtype=tf.float32)

    def setup_inference(self):
        """
        Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)
        """
        if self.discrete:
            self.policy = DiscretePolicy(self.hidden_layer_sizes, output_size=self.ac_dim, activation="tanh")
        else:
            self.policy = ContinuousPolicy(self.hidden_layer_sizes, output_size=self.ac_dim, activation="tanh")

        self.sampled_ac = self.policy(self.obs_ph)
        self.logprob_ac = self.policy.logprob(self.obs_ph, self.acs_ph, name="logprob_ac")
        self.logprob_sampled = self.policy.logprob(self.obs_ph, self.sampled_ac, name="logprob_sampled")
        self.value_targets = tf.placeholder(tf.float32, shape=[None], name="value_fn_targets")

    def setup_loss(self):
        """
        Sets up ppo loss operations for the model.
        """
        # approximate some useful metrics to monitor during training
        self.approx_kl = tf.reduce_mean(self.prev_logprob_ph - self.logprob_ac)
        self.approx_entropy = tf.reduce_mean(-self.logprob_ac)

        # the PPO gradient loss, we use the equivalent simplified version
        prob_ratio = tf.exp(self.logprob_ac - self.prev_logprob_ph)
        loss_values = prob_ratio * self.adv_ph
        clip_values = tf.where(self.adv_ph > 0, (1 + self.clip_ratio) * self.adv_ph, (1 - self.clip_ratio) * self.adv_ph)
        loss = - tf.reduce_mean(tf.minimum(loss_values, clip_values), name="loss")
        optimizer = tf.train.AdamOptimizer(self.policy_learning_rate)
        self.policy_update = optimizer.minimize(loss)

        self.value_fn_prediction = self.value_fn_class(self.obs_ph)
        value_loss = 0.5 * tf.reduce_sum((self.value_fn_prediction - self.value_targets) ** 2,
                                            name="value_fn_loss")
        val_optimizer = tf.train.AdamOptimizer(self.value_fn_learning_rate)
        self.value_fn_update = val_optimizer.minimize(value_loss)

    def setup_graph(self):
        """
        Setup the model, TF graph for inference and loss.
        Call this before training.
        """
        self.setup_placeholders()

        self.setup_inference()
        self.setup_loss()
        self.init_tf()

    def init_tf(self):
        self.sess = tf.keras.backend.get_session()
        self.sess.run(tf.global_variables_initializer())

    def save_model(self, timestep):
        """
        Save current policy model.
        """
        if self.experiments_path != "":
            fpath = os.path.join(self.experiments_path, "models", "model-{}.h5".format(timestep))
            self.policy.save_weights(fpath)

    def load_model(self, model_path=None):
        """
        Load a model.
        If no path passed, loads the latest model.
        """
        if model_path is None:
            # then get latest model
            models_dir = os.path.join(self.experiments_path, "models")
            model_files = os.listdir(models_dir)
            model_number = max([int(f.split(".")[0].split("-")[1]) for f in model_files])
            model_path = os.path.join(models_dir, "model-{}.h5".format(model_number))
        self.policy.load_weights(model_path)

    def sample_trajectories(self, itr):
        """
        Collect paths until we have enough timesteps.
        Returns VPGBuffer() of experience.
        """
        buffer = PPOBuffer()
        while True:
            animate_this_episode = (buffer.length == -1 and itr % self.render_every == 0)
            self.sample_trajectory(buffer, animate_this_episode)
            if buffer.length > self.min_timesteps_per_batch:
                break
            buffer.next()
        return buffer

    def sample_trajectory(self, buffer, render):
        """
        Updates buffer with one episode of experience, rendering the episode if render flag set True.
        """
        ob_ = self.env.reset()
        steps = 0
        while True:
            ob = ob_
            if render:
                self.env.render()
                time.sleep(0.01)
            ac, logprob, val = self.sess.run([self.sampled_ac, self.logprob_sampled, self.value_fn_prediction],
                                             feed_dict={self.obs_ph: np.array([ob])})
            ac = ac[0]
            logprob = logprob[0]
            ob_, rew, done, _ = self.env.step(ac)
            buffer.add(ob, ac, rew, logprob, val)
            steps += 1
            if done:
                # for GAE if we get to terminal state we wrap the episode.
                buffer.done()
                break
            if steps >= self.max_path_length:
                # for GAE pass the estimated value if we finish early.
                buffer.early_stop(self.value_fn_class.predict(ob_))
                break

    def update_parameters(self, obs, acs, rwds, advs, logprobs):
        """
        Update function to call to train policy and value function.
        Returns approx_entropy, approx_kl
        """
        # Optimizing Neural Network Baseline
        for _ in range(self.n_value_fn_updates):
            self.sess.run(self.value_fn_update, feed_dict={self.obs_ph: obs,
                                                           self.value_targets: rwds})
        # compute entropy before update
        approx_entropy = self.sess.run(self.approx_entropy,
                                       feed_dict={self.obs_ph: obs,
                                                  self.acs_ph: acs})
        # Performing the Policy Update
        for _ in range(self.n_policy_updates):
            self.sess.run(self.policy_update, feed_dict={self.obs_ph: obs,
                                                         self.acs_ph: acs,
                                                         self.adv_ph: advs,
                                                         self.prev_logprob_ph: logprobs})
        approx_kl = self.sess.run(self.approx_kl,
                                  feed_dict={self.obs_ph: obs,
                                             self.acs_ph: acs,
                                             self.prev_logprob_ph: logprobs})
        return approx_entropy, approx_kl


def run_model(env, experiments_path, model_path=None, n_episodes=3, **kwargs):
    """
    Run a saved, trained model.
    :param env: environment to run in
    :param experiments_path: the path to the experiments directory, with logs and models
    :param model_path: file path of model to run, if None then latest model in experiments_path is loaded
    :param n_episodes: number of episodes to run
    :param **kwargs: for VPG setup
    """

    ppo = ProximalPolicyOptimisation(env,
                                     experiments_path=experiments_path,
                                     **kwargs)
    ppo.setup_graph()
    ppo.load_model(model_path)

    for i in range(n_episodes):
        buffer = PPOBuffer()
        ppo.sample_trajectory(buffer, True)

        print("Reward: {}".format(sum(buffer.rwds[0])))
