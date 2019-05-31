import tensorflow as tf
import numpy as np
import time
import os
import keras.backend as keras_backend

from src.vpg.utils import VPGBuffer, normalise, GradientBatchTrainer, gaussian_likelihood


class VanillaPolicyGradients:
    def __init__(self, model_fn,
                 env,
                 experiments_path="",
                 discrete=True,
                 learning_rate=5e-3,
                 nn_baseline=False,
                 nn_baseline_fn=None,
                 render_every=25,
                 max_path_length=1000,
                 min_timesteps_per_batch=10000,
                 reward_to_go=True,
                 gamma=0.99,
                 normalise_advantages=True,
                 gradient_batch_size=1000
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

        # make directory to save models
        if self.experiments_path != "":
            save_dir = os.path.join(self.experiments_path, "models")
            if not os.path.exists(save_dir):
                print("\nMade model directory: {}\n".format(save_dir))
                os.makedirs(save_dir)

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
        self.obs_ph = tf.placeholder(shape=[None] + [dim for dim in self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            self.acs_ph = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            self.acs_ph = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)
        self.adv_ph = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        self.old_logprob_ph = tf.placeholder(shape=[None], name="old_logprob", dtype=tf.float32)

    def setup_inference(self, model_outputs):
        """
        Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)
        """
        if self.discrete:
            # here model outputs are the logits, so sample an action
            self.sampled_ac = tf.squeeze(
                tf.random.categorical(model_outputs, 1, dtype=tf.int32, name="sampled_ac"), axis=1)
            # we apply a softmax to get the log probabilities in discrete case
            log_prob = tf.nn.log_softmax(model_outputs)
            indices = tf.stack([tf.range(tf.shape(self.acs_ph)[0]), self.acs_ph], axis=1)
            self.logprob_ac = tf.gather_nd(log_prob, indices, name="logprob_ac")
            indices = tf.stack([tf.range(tf.shape(self.sampled_ac)[0]), self.sampled_ac], axis=1)
            self.logprob_sampled = tf.gather_nd(log_prob, indices, name="logprob_sampled")
        else:
            # sample an action from a Normal(sy_mean, exp(sy_logstd) ** 2)
            # we compute this by transforming a standard normal
            sy_mean = model_outputs
            sy_logstd = tf.get_variable(name="log_std", shape=[self.ac_dim])
            sample_z = tf.random.normal(shape=tf.shape(sy_mean), name="continuous_sample_z")
            self.sampled_ac = sy_mean + tf.exp(sy_logstd) * sample_z
            # formula for log of a gaussian
            self.logprob_ac = gaussian_likelihood(self.acs_ph, sy_mean, sy_logstd)
            self.logprob_sampled = gaussian_likelihood(self.sampled_ac, sy_mean, sy_logstd)

    def setup_loss(self):
        """
        Sets up policy gradient loss operations for the model.
        """
        # approximate some useful metrics to monitor during training
        self.approx_kl = tf.reduce_mean(self.old_logprob_ph - self.logprob_ac)
        self.approx_entropy = tf.reduce_mean(-self.logprob_ac)

        # Loss Function and Training Operation
        loss = - tf.reduce_mean(self.logprob_ac * self.adv_ph, name="loss")

        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        self.policy_batch_trainer = GradientBatchTrainer(loss, self.learning_rate)

        # Optional Baseline
        #
        # Define placeholders for targets, a loss function and an update op for fitting a
        # neural network baseline. These will be used to fit the neural network baseline.
        if self.nn_baseline:
            assert self.nn_baseline_fn is not None, "nn_baseline option requires a nn_baseline_fn"
            model_outputs = self.nn_baseline_fn(self.obs_ph, 1)
            self.baseline_prediction = tf.squeeze(model_outputs)
            # size None because we have vector of length batch size
            self.baseline_targets = tf.placeholder(tf.float32, shape=[None], name="reward_targets")
            baseline_loss = 0.5 * tf.reduce_sum((self.baseline_prediction - self.baseline_targets) ** 2,
                                                name="nn_baseline_loss")
            self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(baseline_loss)
            self.baseline_batch_trainer = GradientBatchTrainer(baseline_loss, self.learning_rate)

    def setup_graph(self):
        """
        Setup the model, TF graph for inference and loss.
        Call this before training.
        """
        self.setup_placeholders()

        model_outputs = self.model_fn(self.obs_ph, self.ac_dim)

        self.setup_inference(model_outputs)
        self.setup_loss()

        self.init_tf()

    def init_tf(self):
        # to change tf Session config, see utils.set_keras_session()
        self.tf_sess = keras_backend.get_session()
        self.tf_sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_model(self, timestep):
        """
        Save current policy model.
        """
        if self.experiments_path != "":
            fpath = os.path.join(self.experiments_path, "models", "model-{}".format(timestep))
            self.saver.save(self.tf_sess, fpath)

    def load_model(self, model_path=None):
        """
        Load a model.
        If no path passed, loads the latest model.
        """
        if model_path is None:
            model_path = tf.train.latest_checkpoint(os.path.join(self.experiments_path, "models"))
        self.saver.restore(self.tf_sess, model_path)

    def sample_trajectories(self, itr):
        """
        Call during training. Calls sample_trajectory to gather enough timesteps for a batch of training.
        :param itr: iteration of training. Used to render frequently to monitor progress.
        """
        # Collect paths until we have enough timesteps
        buffer = VPGBuffer()
        while True:
            render_trajectory = (buffer.length == 0 and (itr % self.render_every == 0))
            self.sample_trajectory(buffer, render_trajectory)
            if buffer.length > self.min_timesteps_per_batch:
                break
            buffer.next()
        return buffer

    def sample_trajectory(self, buffer, render):
        """
        Samples a trajectory from the environment (ie. one episode).
        :param buffer: buffer to store experience in
        :param render: flag whether to render this episode.
        :return:
        """
        ob_ = self.env.reset()
        steps = 0
        while True:
            ob = ob_
            if render:
                self.env.render()
                time.sleep(0.1)
            ac, logprob = self.tf_sess.run([self.sampled_ac, self.logprob_sampled],
                                           feed_dict={self.obs_ph: np.array(ob)[None]})
            ac = ac[0]
            ob_, rew, done, _ = self.env.step(ac)
            buffer.add(ob, ac, rew, logprob[0])
            steps += 1
            if done or steps > self.max_path_length:
                break

    def sum_of_rewards(self, rew_n):
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

            longest_trajectory = max([len(r) for r in rew_n])
            discount = np.triu(
                [[self.gamma ** (i - j) for i in range(longest_trajectory)] for j in range(longest_trajectory)])
            q_n = []
            for re in rew_n:
                # each reward is multiplied
                discounted_re = np.dot(discount[:len(re), : len(re)], re)
                q_n.append(discounted_re)
        else:
            # get discount vector for longest trajectory
            longest_trajectory = max([len(r) for r in rew_n])
            discount = np.array([self.gamma ** i for i in range(longest_trajectory)])
            q_n = []
            for rew in rew_n:
                # np.dot compute the sum of discounted rewards, then we make this into a vector for the
                # full discounted reward case
                disctounted_re = np.ones_like(rew) * np.dot(rew, discount[:len(rew)])
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

            ################################

            # baseline_n = np.zeros_like(q_n)
            # n = int(len(baseline_n) / self.gradient_batch_size)
            # if len(baseline_n) % self.gradient_batch_size != 0: n += 1
            # for i in range(n):
            #     start = i * self.gradient_batch_size
            #     end = (i+1) * self.gradient_batch_size
            #
            #     # prediction from nn baseline
            #     baseline_n[start:end] = self.tf_sess.run(self.baseline_prediction,
            #                                              feed_dict={self.obs_ph: ob_no[start:end]})

            # prediction from nn baseline
            baseline_n = self.tf_sess.run(self.baseline_prediction, feed_dict={self.obs_ph: ob_no})

            # normalise to 0 mean and 1 std
            baseline_n_norm = normalise(baseline_n)
            # set to q mean and std
            q_n_mean = np.mean(q_n)
            q_n_std = np.std(q_n)
            b_n_norm = baseline_n_norm * q_n_std + q_n_mean

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
        # Advantage Normalization
        if self.normalise_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            adv_n = normalise(adv_n)
        return q_n, adv_n

    def update_parameters(self, ob_no, ac_na, q_n, adv_n, logprob_n):
        """
        Update function to call to train policy and (possibly) neural network baseline.
        """
        #====================================================================================#
        #                           ----------PROBLEM 6----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if self.nn_baseline:
            # If a neural network baseline is used, set up the targets and the inputs for the
            # baseline.
            #
            # Fit it to the current batch in order to use for the next iteration. Use the
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the
            # targets to have mean zero and std=1. (Goes with Hint #bl1 in
            # Agent.compute_advantage.)

            # normalise the raw Q-values as per hint above
            target_n = normalise(q_n)
            self.tf_sess.run(self.baseline_update_op, feed_dict={self.obs_ph: ob_no,
                                                         self.baseline_targets: target_n})
            # self.baseline_batch_trainer.train(feed_dict={self.obs_ph: ob_no,
            #                                              self.baseline_targets: target_n},
            #                                   batch_size=self.gradient_batch_size,
            #                                   sess=self.tf_sess)
        # compute entropy before update
        approx_entropy = self.tf_sess.run(self.approx_entropy,
                                          feed_dict={self.obs_ph: ob_no[:self.gradient_batch_size],
                                                     self.acs_ph: ac_na[:self.gradient_batch_size]})
        # Performing the Policy Update
        self.tf_sess.run(self.update_op, feed_dict={self.obs_ph: ob_no,
                                                   self.acs_ph: ac_na,
                                                   self.adv_ph: adv_n})
        # self.policy_batch_trainer.train(feed_dict={self.obs_ph: ob_no,
        #                                            self.acs_ph: ac_na,
        #                                            self.adv_ph: adv_n},
        #                                 batch_size=self.gradient_batch_size,
        #                                 sess=self.tf_sess)
        approx_kl = self.tf_sess.run(self.approx_kl,
                                     feed_dict={self.obs_ph: ob_no[:self.gradient_batch_size],
                                                self.acs_ph: ac_na[:self.gradient_batch_size],
                                                self.old_logprob_ph: logprob_n[:self.gradient_batch_size]})
        return approx_entropy, approx_kl


def run_model(env, model_fn, experiments_path, model_path=None, n_episodes=3, **kwargs):
    """
    Run a saved, trained model.
    :param env: environment to run in
    :param fpath: file path of model to run
    :param n_episodes: number of episodes to run
    :param sleep: time to sleep between steps
    """

    vpg = VanillaPolicyGradients(model_fn,
                                 env,
                                 experiments_path=experiments_path,
                                 **kwargs)
    vpg.setup_graph()
    vpg.load_model(model_path)

    for i in range(n_episodes):
        buffer = VPGBuffer()
        vpg.sample_trajectory(buffer, True)

        print("Reward: {}".format(sum(buffer.rwds[0])))
