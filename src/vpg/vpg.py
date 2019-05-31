import tensorflow as tf
import numpy as np
import time
import os
import keras.backend as keras_backend
from keras import Model, Input
from keras.models import load_model

from src.vpg.utils import VPGBuffer, normalise, gaussian_likelihood


class VanillaPolicyGradients:
    def __init__(self,
                 model_fn,
                 env,
                 experiments_path="",
                 discrete=True,
                 learning_rate=5e-3,
                 nn_baseline=False,
                 nn_baseline_fn=None,
                 render_every=10,
                 max_path_length=1000,
                 min_timesteps_per_batch=10000,
                 reward_to_go=True,
                 gamma=0.99,
                 normalise_advantages=True,
                 gradient_batch_size=100
                 ):

        """
        Run Vanilla Policy Gradients algorithm.

        Parameters
        ----------
        model_fn: src.vpg.models.model
            Model to use for computing the policy, see models.py.
        env: gym.Env
            gym environment to train on.
        experiments_path: string
            path to save models to during training
        discrete: bool
            Whether the environment actions are discrete or continuous
        learning_rate: float
            Learning rate to train with.
        nn_baseline: bool
            Whether to use a neural network baseline when computing advantages
        nn_baseline_fn: src.vpg.models.model
            Model function to compute value baseline, see models.py.
            Ignored if nn_baseline=False.
            Must be set if nn_baseline=True.
        render_every: int
            Render an episode regularly through training to monitor progress
        max_path_length: int
            Max number of timesteps in an episode before stopping.
        min_timesteps_per_batch: int
            Min number of timesteps to gather for use in training updates.
        reward_to_go: bool
            Whether to use reward to go or whole trajectory rewards when discounting and computing advantage.
        gamma: float
            Discount rate
        normalise_advantages: bool
            Whether to normalise advantages.
        gradient_batch_size: int
            To split a batch into mini-batches which the gradient is averaged over to allow larger
            min_timesteps_per_batch than fits into GPU memory in one go.
        """
        self.env = env
        # Is this env continuous, or self.discrete?
        self.discrete = discrete
        self.ob_dim = env.observation_space.shape
        if self.discrete:
            self.ac_dim = env.action_space.n
        else:
            self.ac_dim = env.action_space.shape[0]
        self.model_fn = model_fn

        self.experiments_path = experiments_path

        self.learning_rate = learning_rate
        self.nn_baseline = nn_baseline
        self.nn_baseline_fn = nn_baseline_fn
        self.render_every = render_every
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch
        self.reward_to_go = reward_to_go
        self.gamma = gamma
        self.normalise_advantages = normalise_advantages
        self.gradient_batch_size = gradient_batch_size

    def __str__(self):
        """
        Define string behaviour as key parameters for logging
        """
        to_string = "learning_rate: {}, nn_basline: {}, nn_baseline_fn: {}, " \
                    "max_path_length: {},  min_timesteps_per_batch: {}, " \
                    "reward_to_go: {}, gamma: {}, normalise_advntages: {}".format(self.learning_rate,
                                                                                  self.nn_baseline,
                                                                                  self.nn_baseline_fn,
                                                                                  self.max_path_length,
                                                                                  self.min_timesteps_per_batch,
                                                                                  self.reward_to_go,
                                                                                  self.gamma,
                                                                                  self.normalise_advantages)
        return to_string

    def setup_placeholders(self):
        self.sy_ob_no = tf.placeholder(shape=[None] + [dim for dim in self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            self.sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            self.sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)

        self.sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)

    def setup_inference(self, model_outputs):
        """
        Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)

        Wrapped in Keras Lambdas to build a Keras model
        """
        if self.discrete:
            # here model outputs are the logits
            sy_logits_na = model_outputs
            dist = tf.distributions.Categorical(logits=sy_logits_na, name="discrete_categorical_dist")
            self.sy_sampled_ac = dist.sample(name="discrete_categorical_sample")
            # we apply a softmax to get the log probabilities in discrete case
            self.sy_logprob_n = - tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sy_ac_na, logits=sy_logits_na,
                                                                            name="discrete_action_log_prob")
        else:
            sy_mean = model_outputs
            sy_logstd = tf.get_variable(name="log_std", shape=[self.ac_dim])
            sample_z = tf.random.normal(shape=tf.shape(sy_mean), name="continuous_sample_z")
            self.sy_sampled_ac = sy_mean + tf.exp(sy_logstd) * sample_z
            # formula for log of a gaussian
            sy_std = tf.exp(sy_logstd)
            self.sy_logprob_n = - 0.5 * tf.reduce_sum(((sy_mean - self.sy_ac_na) / sy_std) ** 2, axis=1)

    def setup_loss(self):
        """
        Sets up policy gradient loss operations for the model.
        """
        # Loss Function and Training Operation
        negative_weighted_likelihoods = tf.multiply(self.sy_logprob_n, self.sy_adv_n,
                                                    name="negative_weighted_likelihoods")
        loss = - tf.reduce_mean(negative_weighted_likelihoods, name="loss")
        self.loss_op = loss
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        if self.nn_baseline:
            self.baseline_prediction = tf.squeeze(self.nn_baseline_fn(self.sy_ob_no, 1))
            # size None because we have vector of length batch size
            self.sy_target_n = tf.placeholder(tf.float32, shape=[None], name="reward_targets_nn_V")
            baseline_loss = 0.5 * tf.reduce_sum((self.baseline_prediction - self.sy_target_n) ** 2,
                                                name="nn_baseline_loss")
            self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(baseline_loss)

    def setup_graph(self):
        """
        Setup the model, TF graph for inference and loss.
        Call this before training.
        """
        self.setup_placeholders()

        model_outputs = self.model_fn(self.sy_ob_no, self.ac_dim)
        self.setup_inference(model_outputs)
        self.setup_loss()
        self.init_tf()

    def init_tf(self):
        # to change tf Session config, see utils.set_keras_session()
        self.sess = keras_backend.get_session()
        self.sess.run(tf.global_variables_initializer())

    def save_model(self, timestep):
        """
        Save current policy model.
        """
        # if self.experiments_path != "":
        #     fpath = os.path.join(self.experiments_path, "models", "model-{}.h5".format(timestep))
        #     self.policy_model.save(filepath=fpath)
        pass

    def sample_trajectories(self, itr):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode = (len(paths) == 0 and itr % self.render_every == 0)
            print (itr, animate_this_episode)
            path = self.sample_trajectory(self.env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += len(path['reward'])
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()
        obs, acs, rewards = [], [], []
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
            obs.append(ob)
            # ====================================================================================#
            #                           ----------PROBLEM 3----------
            # ====================================================================================#
            ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: np.array([ob])})
            ac = ac[0]
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)
            rewards.append(rew)
            steps += 1
            if done or steps > self.max_path_length:
                break
        path = {"observation": np.array(obs, dtype=np.float32),
                "reward": np.array(rewards, dtype=np.float32),
                "action": np.array(acs, dtype=np.float32)}
        return path

    def sum_of_rewards(self, re_n):
        """
        Monte Carlo estimation of the Q function.

        Computes discounted sum of rewards for a list of lists of returns each time step.
        Ie. If got reward of one each time step would have
        [[0],
        [0, 1],
        [0, 1, 2],
        ...
        [0, 1, ..., N]]
        """
        if self.reward_to_go:
            # make discount matrix for longest trajectory, N, it's a triangle matrix to offset the timesteps:
            # [gamma^0 gamma^1 ... gamma^N]
            # [ 0 gamma^0 ... gamma^N-1]
            # ...
            # [ 0 0 0 0 ... gamma^0]

            longest_trajectory = max([len(r) for r in re_n])
            discount = np.triu(
                [[self.gamma ** (i - j) for i in range(longest_trajectory)] for j in range(longest_trajectory)])
            q_n = []
            for re in re_n:
                # each reward is multiplied
                discounted_re = np.dot(discount[:len(re), : len(re)], re)
                q_n.append(discounted_re)
        else:
            # get discount vector for longest trajectory
            longest_trajectory = max([len(r) for r in re_n])
            discount = np.array([self.gamma ** i for i in range(longest_trajectory)])
            q_n = []
            for re in re_n:
                # np.dot compute the sum of discounted rewards, then we make this into a vector for the
                # full discounted reward case
                disctounted_re = np.ones_like(re) * np.dot(re, discount[:len(re)])
                q_n.append(disctounted_re)
        q_n = np.hstack(q_n)
        return q_n

    def compute_advantage(self, ob_no, q_n):
        """
        If using neural network baseline then here we compute the estimated values and adjust the sums of rewards
        to compute advantages.
        """
        # Computing Baselines
        if self.nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current batch of Q-values. (Goes with Hint
            # #bl2 in Agent.update_parameters.

            # prediction from nn baseline
            b_n = self.sess.run(self.baseline_prediction, feed_dict={self.sy_ob_no: ob_no})
            # normalise to 0 mean and 1 std
            b_n_norm = normalise(b_n)
            # set to q mean and std
            q_n_mean = np.mean(q_n)
            q_n_std = np.std(q_n)
            b_n_norm = b_n_norm * q_n_std + q_n_mean

            adv_n = q_n - b_n_norm
        else:
            adv_n = q_n.copy()
        return adv_n

    def estimate_return(self, ob_no, rew_n):
        """
        Estimates the returns over a set of trajectories.
        """
        q_n = self.sum_of_rewards(rew_n)
        adv_n = self.compute_advantage(ob_no, q_n)
        if self.normalise_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            adv_n = normalise(adv_n)
        return q_n, adv_n

    def update_parameters(self, ob_no, ac_na, q_n, adv_n):
        """
        Update function to call to train policy and (possibly) neural network baseline.
        """
        if self.nn_baseline:
            target_n = normalise(q_n)

            self.sess.run(self.baseline_update_op, feed_dict={self.sy_ob_no: ob_no, self.sy_target_n: target_n})

        self.sess.run(self.update_op, feed_dict={self.sy_ob_no: ob_no,
                                                 self.sy_ac_na: ac_na,
                                                 self.sy_adv_n: adv_n})


def run_model(env, fpath, n_episodes=3, sleep=0.01):
    """
    Run a saved, trained model.
    :param env: environment to run in
    :param fpath: file path of model to run
    :param n_episodes: number of episodes to run
    :param sleep: time to sleep between steps
    """
    policy_model = load_model(fpath)

    for i in range(n_episodes):

        done = False
        obs = env.reset()
        rwd = 0
        while not done:
            action, _ = policy_model.predict(np.array(obs)[None])

            # step env
            obs, reward, done, info = env.step(action)
            env.render()

            rwd += reward

            time.sleep(sleep)

        print("Reward: {}".format(rwd))
