import tensorflow as tf
import numpy as np
import time
import os
import gym
import logging

from rl_algorithms.src.ppo.utils import PPOBuffer
from rl_algorithms.src.ppo.models import DiscretePolicyFC, DiscretePolicyCNN, ContinuousPolicyFC, ContinuousPolicyCNN
from rl_algorithms.src.common.utils import GradientBatchTrainer, sync_params, sync_experience


class ProximalPolicyOptimisation:
    def __init__(self,
                 env,
                 comm,
                 controller,
                 rank,
                 value_fn,
                 sess_config=None,
                 policy_params={"dense_params": [{"units": 64, "activation": "tanh"}] * 2},
                 experiments_path="",
                 policy_learning_rate=3e-4,
                 value_fn_learning_rate=1e-3,
                 n_policy_updates=50,
                 n_value_updates=50,
                 clip_ratio=0.2,
                 render_every=20,
                 max_path_length=1000,
                 min_timesteps_per_batch=10000,
                 gamma=0.99,
                 gae_lambda=0.97,
                 normalise_advantages=True,
                 entropy_coefficient=1e-4,
                 target_kl=0.01,
                 gradient_batch_size=1000
                 ):

        """
        Run Proximal Policy Optimisation (PPO) algorithm.

        With MPI gathering on the experience and the parameters! ie. Processes perform rollouts and send experience
        to a controller process which performs updates and then communicates parameters to the workers.

        Parameters
        ----------
        env: gym.Env
            gym environment to train on.
        comm:
            an MPI communicator
        controller: int
            rank of the controller process
        rank: int
            rank of this process
        sess_config: tf.ConfigProto
            tf session conifuration, None for default
        policy_params: dict
            if "conv_params" is a key then will add CNN layers before dense.
            uses "dense_params" for the dense NN params.
            See utils/models.py.
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
        value_fn: tf.keras.Model
            Model to compute value function, see models.py. Or None.
        render_every: int
            Render an episode regularly through training to monitor progress
        max_path_length: int
            Max number of timesteps in an episode before stopping.
        min_timesteps_per_batch: int
            Min number of timesteps to gather for use in training updates (total to be split over MPI processes).
        gamma: float
            Discount rate
        gae_lambda: float
            Lambda value for GAE (Between 0 and 1, close to 1)
        normalise_advantages: bool
            Whether to normalise advantages.
        entropy_coefficient: float
            Trades off the entropy term in the PPO loss
        target_kl: float
            KL divergence range acceptable between old and updated policy. Used for early stopping policy updates
        gradient_batch_size: int
            To split a batch into mini-batches which the gradient is averaged over to allow larger
            min_timesteps_per_batch than fits into GPU memory in one go.
        """

        self.env = env
        # Is this env continuous, or self.discrete?
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.ob_dim = env.observation_space.shape
        if self.discrete:
            self.ac_dim = env.action_space.n
        else:
            self.ac_dim = env.action_space.shape[0]

        if value_fn is None:
            print("Value function is None, evaluation only (not training).")

        # MPI settings
        self.comm = comm
        self.controller = controller
        self.rank = rank
        self.sess_config = sess_config

        self.experiments_path = experiments_path

        self.policy_params = policy_params
        self.policy_learning_rate = policy_learning_rate
        self.value_fn_learning_rate = value_fn_learning_rate
        self.n_policy_updates = n_policy_updates
        self.n_value_fn_updates = n_value_updates
        self.clip_ratio = clip_ratio
        self.value_fn = value_fn
        self.render_every = render_every
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalise_advantages = normalise_advantages
        self.entropy_coefficient = entropy_coefficient
        self.target_kl = target_kl
        self.gradient_batch_size = gradient_batch_size

        # make directory to save models
        if self.experiments_path != None:
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
            policy_params: {}
            value_fn_class: {}
            gamma: {}
            n_policy_updates: {}
            n_value_fn_updates: {}
            clip_ratio: {}
            gae_lambda: {}
            entropy_coefficient: {}
            target_kl: {}
            gradient_batch_size: {}
            """.format(
            self.policy_learning_rate,
            self.value_fn_learning_rate,
            self.policy_params,
            self.value_fn,
            self.gamma,
            self.n_policy_updates,
            self.n_value_fn_updates,
            self.clip_ratio,
            self.gae_lambda,
            self.entropy_coefficient,
            self.target_kl,
            self.gradient_batch_size)
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
        if "conv_params" in self.policy_params:
            if self.discrete:
                self.policy = DiscretePolicyCNN(self.policy_params["conv_params"],
                                                self.policy_params["dense_params"],
                                                self.ac_dim)
            else:
                self.policy = ContinuousPolicyCNN(self.policy_params["conv_params"],
                                                  self.policy_params["dense_params"],
                                                  self.ac_dim)
        else:
            if self.discrete:
                self.policy = DiscretePolicyFC(self.policy_params["dense_params"], self.ac_dim)
            else:
                self.policy = ContinuousPolicyFC(self.policy_params["dense_params"], self.ac_dim)

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

        # the PPO gradient loss, we use the equivalent simplified version
        prob_ratio = tf.exp(self.logprob_ac - self.prev_logprob_ph)
        loss_values = prob_ratio * self.adv_ph
        clip_values = tf.where(self.adv_ph > 0, (1 + self.clip_ratio) * self.adv_ph,
                               (1 - self.clip_ratio) * self.adv_ph)
        surrogate_loss = - tf.reduce_mean(tf.minimum(loss_values, clip_values), name="loss")
        # include an entropy term
        probs = tf.exp(self.logprob_ac)
        self.policy_entropy = tf.reduce_mean(-(self.logprob_ac * probs))
        loss = surrogate_loss - self.entropy_coefficient * self.policy_entropy
        self.policy_trainer = GradientBatchTrainer(loss, self.policy_learning_rate, self.policy.trainable_variables)

        if self.value_fn is None:
            # for evaluation only
            self.value_fn_prediction = tf.no_op()
        else:
            self.value_fn_prediction = self.value_fn(self.obs_ph)
            value_loss = 0.5 * tf.reduce_mean((self.value_fn_prediction - self.value_targets) ** 2,
                                             name="value_fn_loss")
            self.value_fn_trainer = GradientBatchTrainer(value_loss, self.value_fn_learning_rate,
                                                         self.value_fn.trainable_variables)

    def setup_graph(self):
        """
        Setup the model, TF graph for inference and loss.
        Call this before training.
        """
        self.setup_placeholders()

        self.setup_inference()
        self.setup_loss()
        self.init_tf()

        # sync the initial params across processes
        if self.comm is not None:
            sync_params(self.policy.variables, self.comm, self.rank, self.controller, self.sess)
            sync_params(self.value_fn.variables, self.comm, self.rank, self.controller, self.sess)

    def init_tf(self):
        self.sess = tf.Session(config=self.sess_config)
        self.sess.run(tf.global_variables_initializer())

    def save_model(self, timestep):
        """
        Save current policy model.
        """
        if self.experiments_path != None:
            fpath = os.path.join(self.experiments_path, "models", "model-{}.h5".format(timestep))
            self.policy.save_weights(fpath)
        else:
            logging.info("Experiments path None in save_model")

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
        Gathers the experience across MPI processes.
        Returns VPGBuffer() of experience. In controller this has full experience. In other
        processes this is empty.
        """
        buffer = PPOBuffer()
        while True:
            animate_this_episode = (buffer.length == -1 and itr % self.render_every == 0)
            self.sample_trajectory(buffer, animate_this_episode)
            # we only get the split of the total data we need per process
            if buffer.length > self.min_timesteps_per_batch / self.comm.Get_size():
                break
            buffer.next()
        # gather the experience in the controller.
        sync_buffer = sync_experience(self.rank, self.controller, self.comm, buffer)
        # empty buffer for non-controller processes
        if sync_buffer is None:
            sync_buffer = PPOBuffer()
        return sync_buffer

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
                buffer.early_stop(self.value_fn.predict(ob_))
                break

    def update_parameters(self, obs, acs, rwds, advs, logprobs):
        """
        Update function to call to train policy and value function.
        Returns policy_entropy, approx_kl
        """
        # perform updates in controller
        if self.rank == self.controller:
            for _ in range(self.n_value_fn_updates):
                self.value_fn_trainer.train(
                    feed_dict={self.obs_ph: obs, self.value_targets: rwds},
                    sess=self.sess, batch_size=self.gradient_batch_size)

            # compute entropy before update
            policy_entropy = self.sess.run(self.policy_entropy,
                                           feed_dict={self.obs_ph: obs[:self.gradient_batch_size],
                                                      self.acs_ph: acs[:self.gradient_batch_size]})
            # Performing the Policy Update
            for _ in range(self.n_policy_updates):
                self.policy_trainer.train(feed_dict={self.obs_ph: obs,
                                                     self.acs_ph: acs,
                                                     self.adv_ph: advs,
                                                     self.prev_logprob_ph: logprobs}, sess=self.sess,
                                          batch_size=self.gradient_batch_size)

                approx_kl = self.sess.run(self.approx_kl,
                                          feed_dict={self.obs_ph: obs[:self.gradient_batch_size],
                                                     self.acs_ph: acs[:self.gradient_batch_size],
                                                     self.prev_logprob_ph: logprobs[:self.gradient_batch_size]})
                if approx_kl > 1.5 * self.target_kl:
                    break
        else:
            # we only log from controller
            approx_kl = None
            policy_entropy = None

        # sync the params across processes
        sync_params(self.value_fn.variables, self.comm, self.rank, self.controller, self.sess)
        sync_params(self.policy.variables, self.comm, self.rank, self.controller, self.sess)

        return policy_entropy, approx_kl

    def gather_logs(self, returns, ep_lens):
        """
        Here the controller is the only one with the full experience already gathered so we only need to return the
        logs in this controller.
        We return None in all processes except controller where logs are actually recorded.
        :return: (gathered) returns, ep_lens
        """
        if self.rank != self.controller:
            returns = None
            ep_lens = None
        return returns, ep_lens


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
                                     None, None, None,  # ignore MPI params
                                     None,  # placeholder, don't need value fn for eval
                                     experiments_path=experiments_path,
                                     **kwargs)
    ppo.setup_graph()
    ppo.load_model(model_path)

    for i in range(n_episodes):
        buffer = PPOBuffer()
        ppo.sample_trajectory(buffer, True)

        print("Reward: {}".format(sum(buffer.rwds[0])))
