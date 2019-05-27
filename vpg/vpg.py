import tensorflow as tf
import numpy as np
import time
import keras.backend as keras_backend


from utils import VPGBuffer, normalise, GradientBatchTrainer


# TODO: comment this class and README
class VanillaPolicyGradients:
    def __init__(self, model_fn, env,
                 discrete=True,
                 learning_rate=5e-3,
                 nn_baseline=False,
                 nn_baseline_fn=None,
                 render_every=10,
                 max_path_length=1000,
                 min_timesteps_per_batch=100000,
                 reward_to_go=True,
                 gamma=0.99,
                 normalise_advantages=True,
                 gradient_batch_size=100
                 ):
        """
        Note that model_fn and (optionally) nn_baseline_fn will be called to construct the TF graph for these operations.
        eg. self.model = model_fn()

        min_timesteps_per_batch is number of timesteps of data to collect before updating parameters.
        gradient_batch_size is to split a batch into mini-batches which the gradient is average over to allow larger
        min_timesteps_per_batch than fits into GPU memory in one go

        :param model_fn:
        :param env:
        :param kwargs:
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

        self.setup_graph()
        self.init_tf()

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
        self.ob_placeholder = tf.placeholder(shape=[None] + [dim for dim in self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            self.ac_placeholder = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            self.ac_placeholder = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)
        self.adv_placeholder = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        self.old_logprob_placeholder = tf.placeholder(shape=[None], name="old_logprob", dtype=tf.float32)

    def setup_inference(self, model_outputs):
        if self.discrete:
            # here model outputs are the logits
            self.sampled_ac = tf.squeeze(tf.multinomial(model_outputs, 1), axis=1)
            return model_outputs
        else:
            sy_mean = model_outputs
            sy_logstd = tf.get_variable(name="log_std", initializer=-0.5 * np.ones(self.ac_dim, dtype=np.float32))
            sample_z = tf.random.normal(shape=tf.shape(sy_mean), name="continuous_sample_z")
            self.sampled_ac = sy_mean + sy_logstd * sample_z
            return sy_mean, sy_logstd

    def gaussian_likelihood(self, x, mu, log_std):
        std = tf.exp(log_std) + 1e-8
        return - 0.5 * tf.reduce_sum(((mu - x) / std) ** 2 + 2 * log_std + np.log(2 * np.pi), axis=1)

    def setup_loss(self, policy_parameters):
        # we apply a softmax to get the log probabilities in discrete case
        if self.discrete:
            model_logits = policy_parameters
            log_prob = tf.nn.log_softmax(model_logits)
            self.logprob_ac = tf.reduce_sum(tf.one_hot(self.ac_placeholder, depth=self.ac_dim) * log_prob, axis=1)
            self.logprob_sampled = tf.reduce_sum(tf.one_hot(self.sampled_ac, depth=self.ac_dim) * log_prob, axis=1)
        else:
            sy_mean, sy_logstd = policy_parameters
            # formula for log of a gaussian
            self.logprob_ac = self.gaussian_likelihood(self.ac_placeholder, sy_mean, sy_logstd)
            self.logprob_sampled = self.gaussian_likelihood(self.sampled_ac, sy_mean, sy_logstd)

        # approximate some useful metrics to monitor during training
        self.approx_kl = tf.reduce_mean(self.old_logprob_placeholder - self.logprob_ac)
        self.approx_entropy = tf.reduce_mean(-self.logprob_ac)

        # Loss Function and Training Operation
        loss = - tf.reduce_mean(self.logprob_ac * self.adv_placeholder, name="loss")

        self.policy_batch_trainer = GradientBatchTrainer(loss, self.learning_rate)

        # Optional Baseline
        #
        # Define placeholders for targets, a loss function and an update op for fitting a
        # neural network baseline. These will be used to fit the neural network baseline.
        if self.nn_baseline:
            assert self.nn_baseline_fn is not None, "nn_baseline option requires a nn_baseline_fn"
            self.baseline_prediction = tf.squeeze(self.nn_baseline_fn(self.ob_placeholder, 1))
            # size None because we have vector of length batch size
            self.baseline_targets = tf.placeholder(tf.float32, shape=[None], name="reward_targets_nn_V")
            baseline_loss = 0.5 * tf.reduce_sum((self.baseline_prediction - self.baseline_targets) ** 2,
                                                name="nn_baseline_loss")
            self.baseline_batch_trainer = GradientBatchTrainer(baseline_loss, self.learning_rate)

    def setup_graph(self):
        self.setup_placeholders()

        model_outputs = self.model_fn(self.ob_placeholder, self.ac_dim)

        # We can sample actions from this action distribution.
        # This will be called in Agent.sample_trajectory() where we generate a rollout.
        policy_parameters = self.setup_inference(model_outputs)

        self.setup_loss(policy_parameters)

    def init_tf(self):
        # to change tf Session config, see utils.set_keras_session()
        self.tf_sess = keras_backend.get_session()
        self.tf_sess.run(tf.global_variables_initializer())

    def sample_trajectories(self, itr):
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
        ob_, rew = self.env.reset(), 0
        steps = 0
        while True:
            ob = ob_
            if render:
                self.env.render()
                time.sleep(0.1)
            ac, logprob = self.tf_sess.run([self.sampled_ac, self.logprob_sampled],
                                        feed_dict={self.ob_placeholder: np.array([ob])})
            ac = ac[0]
            ob_, rew, done, _ = self.env.step(ac)
            buffer.add(ob, ac, rew, logprob[0])
            steps += 1
            if done or steps > self.max_path_length:
                break

    def sum_of_rewards(self, rew_n):
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
        # Computing Baselines
        if self.nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current batch of Q-values. (Goes with Hint
            # #bl2 in Agent.update_parameters.
            adv_n = np.zeros_like(q_n)
            n = int(len(adv_n) / self.gradient_batch_size)
            if len(adv_n) % self.gradient_batch_size != 0: n += 1
            for i in range(n):
                start = i * self.gradient_batch_size
                end = (i+1) * self.gradient_batch_size

                # prediction from nn baseline
                baseline_n = self.tf_sess.run(self.baseline_prediction, feed_dict={self.ob_placeholder: ob_no[start:end]})
                # normalise to 0 mean and 1 std
                baseline_n_norm = normalise(baseline_n)
                # set to q mean and std
                q_n_mean = np.mean(q_n)
                q_n_std = np.std(q_n)
                b_n_norm = baseline_n_norm * q_n_std + q_n_mean

                adv_n[start:end] = b_n_norm

        else:
            adv_n = q_n.copy()
        return adv_n

    def estimate_return(self, ob_no, rew_n):
        q_n = self.sum_of_rewards(rew_n)
        adv_n = self.compute_advantage(ob_no, q_n)
        # Advantage Normalization
        if self.normalise_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            adv_n = normalise(adv_n)
        return q_n, adv_n

    def update_parameters(self, ob_no, ac_na, q_n, adv_n, logprob_n):
        # Optimizing Neural Network Baseline
        if self.nn_baseline:
            # If a neural network baseline is used, set up the targets and the inputs for the
            # baseline.
            target_n = normalise(q_n)
            self.baseline_batch_trainer.train(feed_dict={self.ob_placeholder: ob_no,
                                                         self.baseline_targets: target_n},
                                              batch_size=self.gradient_batch_size,
                                              sess=self.tf_sess)
        # compute entropy before update
        approx_entropy = self.tf_sess.run(self.approx_entropy,
                                       feed_dict={self.ob_placeholder: ob_no[:self.gradient_batch_size],
                                                  self.ac_placeholder: ac_na[:self.gradient_batch_size]})
        # Performing the Policy Update
        self.policy_batch_trainer.train(feed_dict={self.ob_placeholder: ob_no,
                                                   self.ac_placeholder: ac_na,
                                                   self.adv_placeholder: adv_n},
                                        batch_size=self.gradient_batch_size,
                                        sess=self.tf_sess)
        approx_kl = self.tf_sess.run(self.approx_kl,
                                  feed_dict={self.ob_placeholder: ob_no[:self.gradient_batch_size],
                                             self.ac_placeholder: ac_na[:self.gradient_batch_size],
                                             self.old_logprob_placeholder: logprob_n[:self.gradient_batch_size]})
        return approx_entropy, approx_kl
