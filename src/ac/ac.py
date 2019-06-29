import tensorflow as tf
import numpy as np
import time
import os

from src.ac.utils import ACBuffer, normalise
from src.ac.models import DiscretePolicy, ContinuousPolicy, FC_NN


class ActorCrtic:
    def __init__(self,
                 env,
                 critic_model_class=FC_NN,
                 hidden_layer_sizes=[64, 64],
                 experiments_path="",
                 discrete=True,
                 learning_rate_actor=5e-3,
                 learning_rate_critic=5e-3,
                 size_actor=64,
                 size_critic=64,
                 n_layers_actor=2,
                 n_layers_critic=2,
                 num_target_updates=10,
                 num_grad_steps_per_target_update=10,
                 render_every=20,
                 max_path_length=1000,
                 min_timesteps_per_batch=10000,
                 reward_to_go=True,
                 gamma=0.99,
                 normalise_advantages=True,
                 gradient_batch_size=1000
                 ):

        """
        Run Actor Critic algorithm.

        Parameters
        ----------
        env: gym.Env
            gym environment to train on.
        critic_model_class: tf.keras.Model
            Model class to use for critic, should subclass tf.keras.Model see src.common.models.py.
        hidden_layer_sizes: list
            List of ints for the number of units to have in the hidden layers of the policy and the critic
        experiments_path: string
            path to save models to during training
        discrete: bool
            Whether the environment actions are discrete or continuous
        learning_rate_actor: float
            Learning rate to train actor with.
        size_actor: int
            size for actor hidden layers (number of units).
        n_layers_actor: int
            number of hidden layers in actor
        learning_rate_actor: float
            Learning rate to train critic with.
        size_actor: int
            size for critic hidden layers (number of units).
        n_layers_actor: int
            number of hidden layers in critic
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
        num_target_updates: int
            number of target updates to critic, where each target update consists of num_grad_steps_per_target_update
            update steps, recomputing the targets each target update.
        num_grad_steps_per_target_update:
            number of gradient steps per target update for critic.
        """
        self.env = env
        # Is this env continuous, or self.discrete?
        self.discrete = discrete
        self.ob_dim = env.observation_space.shape
        if self.discrete:
            self.ac_dim = env.action_space.n
        else:
            self.ac_dim = env.action_space.shape[0]
        self.critic_model_class = critic_model_class

        self.experiments_path = experiments_path

        # actor params
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_actor = learning_rate_actor
        self.size_actor = size_actor
        self.n_layers_actor = n_layers_actor
        # critic params
        self.learning_rate_critic = learning_rate_critic
        self.size_critic = size_critic
        self.n_layers_critic = n_layers_critic
        # training params
        self.num_grad_steps_per_target_update = num_grad_steps_per_target_update
        self.num_target_updates = num_target_updates
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
        to_string = """
                    critic_model_class: {}
                    hidden_layer_sizes: {}
                    discrete: {}
                    learning_rate_actor: {}
                    learning_rate_critic: {}
                    size_actor: {}, size_critic: {}
                    n_layers_actor: {}, n_layers_critic: {} 
                    num_grad_steps_per_target_update: {}
                    num_target_updates: {}
                    max_path_length: {}
                    min_timesteps_per_batch: {}
                    reward_to_go: {}
                    gamma: {}
                    normalise_advntages: {}""".format(self.critic_model_class,
                                                      self.hidden_layer_sizes,
                                                      self.discrete,
                                                      self.learning_rate_actor,
                                                      self.learning_rate_critic,
                                                      self.size_actor,
                                                      self.size_critic,
                                                      self.n_layers_actor,
                                                      self.n_layers_critic,
                                                      self.num_grad_steps_per_target_update,
                                                      self.num_target_updates,
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

    def setup_loss(self):
        """
        Sets up loss operations for the actor and critic.
        """
        # approximate some useful metrics to monitor during training
        self.approx_kl = tf.reduce_mean(self.prev_logprob_ph - self.logprob_ac)
        self.approx_entropy = tf.reduce_mean(-self.logprob_ac)

        # the policy gradient loss
        actor_loss = - tf.reduce_mean(self.logprob_ac * self.adv_ph, name="loss")
        self.actor_update_op = tf.train.AdamOptimizer(self.learning_rate_actor).minimize(actor_loss)

        # define the critic
        self.critic_model = self.critic_model_class(self.hidden_layer_sizes, 1)
        self.critic_prediction = self.critic_model(self.obs_ph)
        self.critic_target_ph = tf.placeholder(shape=[None], name="critic_target", dtype=tf.float32)
        critic_loss = tf.losses.mean_squared_error(self.critic_target_ph, self.critic_prediction)
        self.critic_update_op = tf.train.AdamOptimizer(self.learning_rate_critic).minimize(critic_loss)

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
        Returns ACBuffer() of experience.
        """
        buffer = ACBuffer()
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
            steps += 1
            if done or steps >= self.max_path_length:
                buffer.add(ob, ac, rew, ob_, 1, logprob)  # terminal is 1 if finished
                break

            buffer.add(ob, ac, rew, ob_, 0, logprob)  # terminal is 0 if not done

    def estimate_advantage(self, ob, next_ob, rew, terminal_n):
        """
            Estimates the advantage function value for each timestep.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories

            arguments:
                ob: shape: (sum_of_path_lengths, ob_dim)
                next_ob: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                rew: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
        """
        V_no = self.sess.run(self.critic_prediction, feed_dict={self.obs_ph: next_ob})
        Q_n = rew + self.gamma * V_no * (1 - terminal_n)  # mask if done
        V_n = self.sess.run(self.critic_prediction, feed_dict={self.obs_ph: ob})
        adv_n = Q_n - V_n

        if self.normalise_advantages:
            adv_n = normalise(adv_n)
        return adv_n

    def update_critic(self, ob, next_ob, rew, terminal_n):
        """
            Use bootstrapped target values to update the critic

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob: shape: (sum_of_path_lengths, ob_dim)
                next_ob: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                rew: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
        """
        for _ in range(self.num_target_updates):
            V_no = self.sess.run(self.critic_prediction, feed_dict={self.obs_ph: next_ob})
            targets_n = rew + self.gamma * V_no * (1 - terminal_n)  # mask if done
            for _ in range(self.num_grad_steps_per_target_update):
                self.sess.run(self.critic_update_op, feed_dict={self.critic_target_ph: targets_n,
                                                                self.obs_ph: ob})

    def update_actor(self, ob_no, ac_na, adv_n):
        """
            Update the parameters of the policy.

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: shape: (sum_of_path_lengths).
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths
        """
        self.sess.run(self.actor_update_op,
                      feed_dict={self.obs_ph: ob_no, self.acs_ph: ac_na, self.adv_ph: adv_n})

    def update_parameters(self, obs, next_obs, rews, terminal_n, acs):
        """
        Main update function to call to train actor and critic.
        Returns approx_entropy, approx_kl
        """
        self.update_critic(obs, next_obs, rews, terminal_n)
        adv_n = self.estimate_advantage(obs, next_obs, rews, terminal_n)
        self.update_actor(obs, acs, adv_n)

    def training_metrics(self, obs, acs, logprobs):
        """
        Logging function
        Returns approx_entropy, approx_kl
        """
        # compute entropy before update
        approx_entropy = self.sess.run(self.approx_entropy,
                                       feed_dict={self.obs_ph: obs[:self.gradient_batch_size],
                                                  self.acs_ph: acs[:self.gradient_batch_size]})
        approx_kl = self.sess.run(self.approx_kl,
                                  feed_dict={self.obs_ph: obs[:self.gradient_batch_size],
                                             self.acs_ph: acs[:self.gradient_batch_size],
                                             self.prev_logprob_ph: logprobs[:self.gradient_batch_size]})
        return approx_entropy, approx_kl


def run_model(env, experiments_path, model_path=None, n_episodes=3, **kwargs):
    """
    Run a saved, trained model.
    :param env: environment to run in
    :param model_fn: the model function to setup the policy with
    :param experiments_path: the path to the experiments directory, with logs and models
    :param model_path: file path of model to run, if None then latest model in experiments_path is loaded
    :param n_episodes: number of episodes to run
    :param **kwargs: for AC setup
    """

    actrcrtc = ActorCrtic(env,
                          experiments_path=experiments_path,
                          **kwargs)
    actrcrtc.setup_graph()
    actrcrtc.load_model(model_path)

    for i in range(n_episodes):
        buffer = ACBuffer()
        actrcrtc.sample_trajectory(buffer, True)

        print("Reward: {}".format(sum(buffer.rwds[0])))
