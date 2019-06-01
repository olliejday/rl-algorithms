import tensorflow as tf
import numpy as np
import time
import os
import keras.backend as keras_backend

from src.vpg.utils import VPGBuffer, GradientBatchTrainer, normalise, gaussian_log_likelihood, gather_nd


class VanillaPolicyGradients:
    def __init__(self,
                 model_fn,
                 env,
                 experiments_path="",
                 discrete=True,
                 learning_rate=5e-3,
                 nn_baseline=False,
                 nn_baseline_fn=None,
                 render_every=20,
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
                print("Made model directory: {}".format(save_dir))
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
        # for approx KL
        self.prev_logprob_ph = tf.placeholder(shape=[None], name="prev_logprob", dtype=tf.float32)

    def setup_inference(self, model_outputs):
        """
        Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)

        Wrapped in Keras Lambdas to build a Keras model
        """
        if self.discrete:
            # here model outputs are the logits
            self.sampled_ac = tf.squeeze(tf.random.categorical(model_outputs, 1, name="sampled_ac", dtype=tf.int32),
                                         axis=1)
            # we apply a softmax to get the log probabilities in discrete case
            logprob = tf.nn.log_softmax(model_outputs)
            self.logprob_ac = gather_nd(logprob, self.acs_ph, name="logprob_ac")
            self.logprob_sampled = gather_nd(logprob, self.sampled_ac, name="logprob_sampled")
        else:
            sy_mean = model_outputs
            sy_logstd = tf.get_variable(name="log_std", shape=[self.ac_dim])
            # get sample by sampling from standard normal then transforming
            sample_z = tf.random.normal(shape=tf.shape(sy_mean), name="continuous_sample_z")
            self.sampled_ac = sy_mean + tf.exp(sy_logstd) * sample_z
            # log probability
            self.logprob_ac = gaussian_log_likelihood(self.acs_ph, sy_mean, sy_logstd)
            self.logprob_sampled = gaussian_log_likelihood(self.sampled_ac, sy_mean, sy_logstd)

    def setup_loss(self):
        """
        Sets up policy gradient loss operations for the model.
        """
        # approximate some useful metrics to monitor during training
        self.approx_kl = tf.reduce_mean(self.prev_logprob_ph - self.logprob_ac)
        self.approx_entropy = tf.reduce_mean(-self.logprob_ac)

        # the policy gradient loss
        loss = - tf.reduce_mean(self.logprob_ac * self.adv_ph, name="loss")
        # self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        self.policy_batch_trainer = GradientBatchTrainer(loss, self.learning_rate)


        if self.nn_baseline:
            self.baseline_prediction = tf.squeeze(self.nn_baseline_fn(self.obs_ph, 1))
            # size None because we have vector of length batch size
            self.baseline_targets_ph = tf.placeholder(tf.float32, shape=[None], name="reward_targets_nn_V")
            # baseline loss on true reward targets
            baseline_loss = 0.5 * tf.reduce_sum((self.baseline_prediction - self.baseline_targets_ph) ** 2,
                                                name="nn_baseline_loss")
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
        self.sess = keras_backend.get_session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_model(self, timestep):
        """
        Save current policy model.
        """
        if self.experiments_path != "":
            fpath = os.path.join(self.experiments_path, "models", "model-{}".format(timestep))
            self.saver.save(self.sess, fpath)

    def load_model(self, model_path=None):
        """
        Load a model.
        If no path passed, loads the latest model.
        """
        if model_path is None:
            model_path = tf.train.latest_checkpoint(os.path.join(self.experiments_path, "models"))
        self.saver.restore(self.sess, model_path)

    def sample_trajectories(self, itr):
        """
        Collect paths until we have enough timesteps.
        Returns VPGBuffer() of experience.
        """
        buffer = VPGBuffer()
        while True:
            animate_this_episode = (buffer.length == -1 and itr % self.render_every == 0)
            self.sample_trajectory(buffer, animate_this_episode)
            if buffer.length > self.min_timesteps_per_batch:
                break
        return buffer

    def sample_trajectory(self, buffer, render):
        """
        Updates buffer with one episode of experience, rendering the episode if render flag set True.
        """
        buffer.next()
        ob_ = self.env.reset()
        steps = 0
        while True:
            ob = ob_
            if render:
                self.env.render()
                time.sleep(0.01)
            ac, logprob = self.sess.run([self.sampled_ac, self.logprob_sampled],
                                        feed_dict={self.obs_ph: np.array([ob])})
            ac = ac[0]
            logprob = logprob[0]
            ob_, rew, done, _ = self.env.step(ac)
            buffer.add(ob, ac, rew, logprob)
            steps += 1
            if done or steps >= self.max_path_length:
                break

    def sum_of_rewards(self, rwds):
        """
        Monte Carlo estimation of the Q function.

        Computes discounted sum of rewards for a list of lists of returns each trajectory.

        Reward to go :
            In this case use reward to go, ie. each timestep's return is the sum of discounted rewards from
            then until end.
        Otherwise:
            In this case use full sum of discounted rewards as the return for each timestep

        Returns qs (discounted reward sequences)
        """
        if self.reward_to_go:
            # make discount matrix for longest trajectory, N, it's a triangle matrix to offset the timesteps:
            # [gamma^0 gamma^1 ... gamma^N]
            # [ 0 gamma^0 ... gamma^N-1]
            # ...
            # [ 0 0 0 0 ... gamma^0]

            longest_trajectory = max([len(r) for r in rwds])
            discount = np.triu(
                [[self.gamma ** (i - j) for i in range(longest_trajectory)] for j in range(longest_trajectory)])
            qs = []
            for re in rwds:
                # each reward is multiplied
                discounted_re = np.dot(discount[:len(re), : len(re)], re)
                qs.append(discounted_re)
        else:
            # get discount vector for longest trajectory
            longest_trajectory = max([len(r) for r in rwds])
            discount = np.array([self.gamma ** i for i in range(longest_trajectory)])
            qs = []
            for re in rwds:
                # np.dot compute the sum of discounted rewards, then we make this into a vector for the
                # full discounted reward case
                disctounted_re = np.ones_like(re) * np.dot(re, discount[:len(re)])
                qs.append(disctounted_re)
        qs = np.hstack(qs)
        return qs

    def compute_advantage(self, obs, qs):
        """
        If using neural network baseline then here we compute the estimated values and adjust the sums of rewards
        to compute advantages.

        Returns advs (advantage estimates)
        """
        # Computing Baselines
        if self.nn_baseline:
            # prediction from nn baseline
            baseline_preds = np.zeros_like(qs)
            n = int(len(baseline_preds) / self.gradient_batch_size)
            if len(baseline_preds) % self.gradient_batch_size != 0: n += 1
            for i in range(n):
                start = i * self.gradient_batch_size
                end = (i+1) * self.gradient_batch_size

                # prediction from nn baseline
                baseline_preds[start:end] = self.sess.run(self.baseline_prediction,
                                                      feed_dict={self.obs_ph: obs[start:end]})
            # normalise to 0 mean and 1 std
            bn_norm = normalise(baseline_preds)
            # set to q mean and std
            qs_mean = np.mean(qs)
            qs_std = np.std(qs)
            bn_norm = bn_norm * qs_std + qs_mean

            advs = qs - bn_norm
        else:
            advs = qs.copy()
        return advs

    def estimate_return(self, obs, rews):
        """
        Estimates the returns over a set of trajectories.
        Can normalise advantaged to reduce variance.

        Returns q_n, adv_n (disctounted sum of rewards, advantaged estimates)
        """
        q_n = self.sum_of_rewards(rews)
        adv_n = self.compute_advantage(obs, q_n)
        if self.normalise_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            adv_n = normalise(adv_n)
        return q_n, adv_n

    def update_parameters(self, obs, acs, qs, advs, logprobs):
        """
        Update function to call to train policy and (possibly) neural network baseline.
        Returns approx_entropy, approx_kl
        """
        # Optimizing Neural Network Baseline
        if self.nn_baseline:
            # If a neural network baseline is used, set up the targets and the inputs for the
            # baseline.
            target_n = normalise(qs)
            self.baseline_batch_trainer.train(feed_dict={self.obs_ph: obs,
                                                         self.baseline_targets_ph: target_n},
                                              batch_size=self.gradient_batch_size,
                                              sess=self.sess)
        # compute entropy before update
        approx_entropy = self.sess.run(self.approx_entropy,
                                       feed_dict={self.obs_ph: obs[:self.gradient_batch_size],
                                                  self.acs_ph: acs[:self.gradient_batch_size]})
        # Performing the Policy Update
        self.policy_batch_trainer.train(feed_dict={self.obs_ph: obs,
                                                   self.acs_ph: acs,
                                                   self.adv_ph: advs},
                                        batch_size=self.gradient_batch_size,
                                        sess=self.sess)
        # self.sess.run(self.update_op, feed_dict={self.obs_ph: obs,
        #                                            self.acs_ph: acs,
        #                                            self.adv_ph: advs})
        approx_kl = self.sess.run(self.approx_kl,
                                     feed_dict={self.obs_ph: obs[:self.gradient_batch_size],
                                                self.acs_ph: acs[:self.gradient_batch_size],
                                                self.prev_logprob_ph: logprobs[:self.gradient_batch_size]})
        return approx_entropy, approx_kl


def run_model(env, model_fn, experiments_path, model_path=None, n_episodes=3, **kwargs):
    """
    Run a saved, trained model.
    :param env: environment to run in
    :param model_fn: the model function to setup the policy with
    :param experiments_path: the path to the experiments directory, with logs and models
    :param model_path: file path of model to run, if None then latest model in experiments_path is loaded
    :param n_episodes: number of episodes to run
    :param **kwargs: for VPG setup
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
