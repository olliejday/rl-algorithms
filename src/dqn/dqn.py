import time
import gym.spaces
import numpy as np
import os
import tensorflow as tf

from src.dqn.utils import LinearSchedule, huber_loss, get_wrapper_by_name, DQNReplayBuffer, ConstantSchedule


class DQN():
    def __init__(
            self,
            env,
            model_class,
            optimizer_spec,
            exploration=LinearSchedule(1000000, 0.1),
            replay_buffer_size=1000000,
            batch_size=32,
            gamma=0.99,
            learning_starts=50000,
            learning_freq=4,
            frame_history_len=4,
            target_update_freq=10000,
            grad_norm_clipping=10,
            delta=1.0,
            save_every=1e7,
            experiments_dir="",
            double_q=True,
            log_every_n_steps=1e5,
            integer_observations=True,
            render=False):
        """
        Run Deep Q-learning algorithm.

        All schedules are w.r.t. total number of steps taken in the environment.

        Parameters
        ----------
        env: gym.Env
            gym environment to train on.
        model_class: tf.keras.Model
            Model to use for computing the q function.
            The model should be a subclass of tf.keras.Model so that we can setup layer sharing, see models.py
        optimizer_spec: src.dqn.utils.OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate schedule
            for the optimizer
            If None, we assume evaluation only (no training)
        exploration: src.dqn.utils.Schedule
            schedule for probability of chosing random action.
        replay_buffer_size: int
            How many memories to store in the replay buffer.
        batch_size: int
            How many transitions to sample each time experience is replayed.
        gamma: float
            Discount Factor
        learning_starts: int
            After how many environment steps to start replaying experiences
        learning_freq: int
            How many steps of environment to take between every experience replay
        frame_history_len: int
            How many past frames to include as input to the model.
        target_update_freq: int
            How many experience replay rounds (not steps!) to perform between
            each update to the target Q network
        grad_norm_clipping: float or None
            If not None gradients' norms are clipped to this value.
        delta: float
            for huber loss
        double_q: bool
            If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
        save_every: int
            saver a model every N timesteps
        log_every_n_steps: int
            print and save to file logs of training metrics every N timesteps
        experiments_dir: str
            where to save models and logs to
        integer_observations: bool
            for replay buffer whether to store integers or floats
        render: bool
            whether to render the environment when reporting logs or not
        """

        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete

        # environment parameters
        self.env = env
        self.ob_dim = env.observation_space.shape
        self.ac_dim = env.action_space.n
        self.integer_observations = integer_observations

        # have separate weights for target and q_model
        self.q_model_class = model_class
        self.target_model_class = model_class

        # model parameters
        self.target_update_freq = target_update_freq
        self.optimizer_spec = optimizer_spec
        self.batch_size = batch_size
        self.learning_freq = learning_freq
        self.learning_starts = learning_starts
        self.gamma = gamma
        self.exploration = exploration
        self.grad_norm_clipping = grad_norm_clipping
        self.delta = delta
        self.double_q = double_q
        self.save_every = save_every
        self.experiments_dir = experiments_dir
        self.replay_buffer_size = replay_buffer_size
        self.frame_history_len = frame_history_len

        # make directory to save models
        if self.experiments_dir != "":
            save_dir = os.path.join(self.experiments_dir, "models")
            if not os.path.exists(save_dir):
                print("Made model directory: {}".format(save_dir))
                os.makedirs(save_dir)

        # for rendering during training
        self.render = render
        self.render_flag = False

        self.last_obs = self.env.reset()
        self.num_param_updates = 0
        self.log_every_n_steps = log_every_n_steps

        self.start_time = None
        self.t = 0

    def __str__(self):
        """
        For printing the parameters of an experiment to logs.
        """
        to_str = """
        batch_size: {}, 
        gamma: {}, 
        learning_starts: {}, 
        learning_freq: {},
        frame_history_len: {}, 
        target_update_freq: {}, 
        grad_norm_clipping: {}, 
        double_q: {}, 
        replay_buffer_size: {}, 
        target_update_freq: {}""".format(
            self.batch_size,
            self.gamma,
            self.learning_starts,
            self.learning_freq,
            self.frame_history_len,
            self.target_update_freq,
            self.grad_norm_clipping,
            self.double_q,
            self.replay_buffer_size,
            self.target_update_freq)
        return to_str

    def setup_placeholders(self):
        # set up placeholders
        # placeholder for current observation (or state)
        if len(self.env.observation_space.shape) == 1:
            # low dimensional observations
            input_shape = self.ob_dim
        else:
            # high dimensional observations eg. images
            img_h, img_w, img_c = self.ob_dim
            input_shape = (img_h, img_w, self.frame_history_len * img_c)
        # casting to float on GPU ensures lower data transfer times.
        if self.integer_observations:
            self.ob_placeholder = tf.placeholder(shape=[None] + [dim for dim in input_shape], name="observation",
                                                 dtype=tf.uint8)
            self.next_ob_placeholder = tf.placeholder(shape=[None] + [dim for dim in input_shape], name="next_ob",
                                                      dtype=tf.uint8)
        else:
            self.ob_placeholder = tf.placeholder(shape=[None] + [dim for dim in input_shape], name="observation",
                                                 dtype=tf.float32)
            self.next_ob_placeholder = tf.placeholder(shape=[None] + [dim for dim in input_shape], name="next_ob",
                                                      dtype=tf.float32)
        self.ac_placeholder = tf.placeholder(shape=[None], name="action", dtype=tf.int32)
        self.rew_placeholder = tf.placeholder(shape=[None], name="reward", dtype=tf.float32)
        self.done_mask_ph = tf.placeholder(shape=[None], name="done_mask", dtype=tf.float32)
        self.learning_rate = tf.placeholder(shape=(), name="learning_rate", dtype=tf.float32)

    def setup_inference(self):
        # we init the model class to setup the model. It can then be called as a function on an input placeholder to
        # return the outputs
        self.q_model = self.q_model_class(self.ac_dim, integer_observations=self.integer_observations, env=self.env)
        self.q_model_ob = self.q_model(self.ob_placeholder)
        self.q_model_next_ob = self.q_model(self.next_ob_placeholder)

        self.target_model = self.target_model_class(self.ac_dim, integer_observations=self.integer_observations,
                                                    env=self.env)
        self.target_model_outputs = self.target_model(self.next_ob_placeholder)
        self.target_q_func_vars = self.target_model.trainable_weights

    def setup_loss(self):
        """
        Define the loss and training operations.
        """
        # Define the loss tf operation
        if self.double_q:  # double DQN, use current network to evaluate actions, max_value_action = argmax_{a'} Q_{phi}(s', a')
            max_value_action = tf.argmax(self.q_model_next_ob, axis=1, name="max_value_action", output_type=tf.int32)
        else:  # vanilla DQN, use target network to evaluate actions, max_value_action = argmax_{a'} Q_{phi'}(s', a')
            max_value_action = tf.argmax(self.target_model_outputs, axis=1, name="max_value_action",
                                         output_type=tf.int32)

        # indexing the max valued actions into the target Q values
        indices = tf.stack([tf.range(tf.shape(max_value_action)[0]), max_value_action], axis=1)
        max_Q_value = tf.gather_nd(self.target_model_outputs, indices, name="max_Q_value")

        # y = r + (1 - done) * gamma * Q_{phi'}(s', max_value_action)
        target_values = self.rew_placeholder + (1 - self.done_mask_ph) * self.gamma * max_Q_value

        # indexing the actions into the Q values
        indices = tf.stack([tf.range(tf.shape(self.q_model_ob)[0]), self.ac_placeholder], axis=1)
        action_taken_Q_vals = tf.gather_nd(self.q_model_ob, indices, name="action_Q_value")

        loss = target_values - action_taken_Q_vals
        self.total_error = tf.reduce_mean(huber_loss(loss, self.delta))

        ######

        # construct optimization op (with gradient clipping)
        optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
        gradients = optimizer.compute_gradients(self.total_error, var_list=self.q_model.trainable_weights)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var)
        self.train_fn = optimizer.apply_gradients(gradients)

        # update_target_fn will be called periodically to copy Q network to target Q network
        def update_target_fn():
            self.target_model.set_weights(self.q_model.get_weights())

        self.update_target_fn = update_target_fn

    def setup_replay_buffer(self):
        # construct the replay buffer
        self.replay_buffer = DQNReplayBuffer(self.replay_buffer_size, self.frame_history_len, self.integer_observations)
        self.replay_buffer_idx = None

    def setup_graph(self):
        """
        Setup the model, TF graph for inference and loss.
        Call this before training.
        """
        self.setup_placeholders()
        self.setup_inference()
        if self.optimizer_spec is not None:
            self.setup_loss()
        self.setup_replay_buffer()
        self.init_tf()

    def init_tf(self):
        # to change tf Session config, see utils.set_keras_session()
        self.tf_sess = tf.keras.backend.get_session()
        self.tf_sess.run(tf.global_variables_initializer())

    def step_env(self):
        """
        Take one step in the environment.
        Epsilon greedy exploration
        """

        # store the latest observation
        idx = self.replay_buffer.store_frame(self.last_obs)

        # encode the current observation, [None] adds a dimension of 1 for the batch
        encoded_observation = self.replay_buffer.encode_recent_observation()[None]

        # evaluate model on this observation
        if self.t > self.learning_starts:
            # epsilon greedy
            # exploration step
            if np.random.rand() < self.exploration.value(self.t):
                action = np.random.choice(self.ac_dim)
            # follow policy
            else:
                Q_values = self.q_model.predict(encoded_observation)
                action = np.argmax(Q_values)
        # if model not initialised take a random action
        else:
            action = np.random.choice(self.ac_dim)

        # render the env
        if self.render_flag and self.render:
            self.env.render()
            time.sleep(0.01)

        # step env
        obs, reward, done, info = self.env.step(action)

        if done:
            self.render_flag = False  # only render one episode
            obs = self.env.reset()

        # store this step
        self.replay_buffer.store_effect(idx, action, reward, done)
        self.last_obs = obs

        return done

    def update_model(self):
        """
        The model update call.
        Minimises loss using the optimizer spec.
        Updates target model every target_update_freq timesteps.
        """
        if (self.t > self.learning_starts and
                self.t % self.learning_freq == 0 and
                self.replay_buffer.can_sample(self.batch_size)):
            # from buffer
            obs_batch, acs_batch, rews_batch, next_obs_batch, done_mask = self.replay_buffer.sample(self.batch_size)

            lr_t = self.optimizer_spec.lr_schedule.value(self.t)
            # train step
            self.tf_sess.run(self.train_fn, feed_dict={self.ob_placeholder: obs_batch,
                                                       self.ac_placeholder: acs_batch,
                                                       self.rew_placeholder: rews_batch,
                                                       self.next_ob_placeholder: next_obs_batch,
                                                       self.done_mask_ph: done_mask,
                                                       self.learning_rate: lr_t})

            # update target network
            # we want to keep to a frequency of updates, so if we are less that that frequency we update
            if self.num_param_updates % self.target_update_freq == 0:
                self.update_target_fn()

            # count number of experience replay updates
            self.num_param_updates += 1

        self.t += 1

    def log_progress(self):
        """
        Saves model if a saving timestep
        Returns None if not a logging step,
        Returns (timestep, episode rewards, episode lengths and exploration parameter) on logging steps
        """
        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
        episode_lengths = get_wrapper_by_name(self.env, "Monitor").get_episode_lengths()

        # save model
        if self.t % self.save_every == 0:
            self.save_model()
        if self.t % self.log_every_n_steps == 0:
            # render next episode
            self.render_flag = True
            return self.t, episode_rewards, episode_lengths, self.exploration.value(self.t)

        return None

    def save_model(self):
        """
        Save current DQN Q-network model.
        """
        if self.experiments_dir != "":
            fpath = os.path.join(self.experiments_dir, "models", "model-{}.h5".format(self.t))
            self.q_model.save_weights(fpath)

    def load_model(self, model_path):
        """
        Load a model.
        """
        self.q_model.load_weights(model_path)


def run_model(env, model_class, model_path, n_episodes=3, **kwargs):
    """
    Run a saved, trained model.
    :param env: environment to run in
    :param model_class: model class to setup DQN
    :param model_path: path of model to load
    :param n_episodes: number of episodes to run
    :param sleep: time to sleep between steps
    :param kwargs: for DQN()
    """

    dqn = DQN(env,
              model_class,
              optimizer_spec=None,  # None means no training
              exploration=ConstantSchedule(0),  # No exploration
              replay_buffer_size=1000,  # Don't need a massive buffer for evaluation
              render=True,  # we want to render the evaluation
              learning_starts=-1,  # don't need the initial random actions
              **kwargs)

    dqn.setup_graph()

    dqn.load_model(model_path)

    for i in range(n_episodes):
        done = False
        while not done:
            dqn.render_flag = True
            done = dqn.step_env()
        print("Reward: {}".format(get_wrapper_by_name(env, "Monitor").get_episode_rewards()[-1]))
