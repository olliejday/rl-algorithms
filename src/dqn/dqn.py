import time
import gym.spaces
import numpy as np
import tensorflow as tf
import os
import keras.backend as keras_backend
from keras import Model
from keras.layers import Input, Lambda
from keras.models import load_model

from utils import LinearSchedule, huber_loss, get_wrapper_by_name, ReplayBuffer


# TODO: comment and readme this class
# TODO: comment the model class then call as fn thing
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
            experiments_path="",
            double_q=True,
            log_every_n_steps=1e5,
            integer_observations=True,):

        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete

        self.env = env
        self.ob_dim = env.observation_space.shape
        self.ac_dim = env.action_space.n
        self.integer_observations = integer_observations

        # the model should be a class so that we can setup keras layer sharing
        self.q_model_class = model_class
        self.target_model_class = model_class

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
        self.experiments_path = experiments_path
        self.replay_buffer_size = replay_buffer_size
        self.frame_history_len = frame_history_len

        # make models dir
        if self.experiments_path != "":
            save_dir = os.path.join(self.experiments_path, "models")
            if not os.path.exists(save_dir):
                print("\nMade model directory: {}\n".format(save_dir))
                os.makedirs(save_dir)

        # for rendering during training
        self.render_flag = True

        self.last_obs = self.env.reset()
        self.num_param_updates = 0
        self.log_every_n_steps = log_every_n_steps

        self.start_time = None
        self.t = 0

    def __str__(self):
        # for logging

        to_str = "batch_size: {}, gamma: {}, learning_starts: {}, learning_freq: {}, " \
                 "frame_history_len: {}, target_update_freq: {}, grad_norm_clipping: {}, "
        "double_q: {}, replay_buffer_size: {}, target_update_freq: {}".format(self.batch_size,
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
            self.ob_placeholder = Input(shape=[dim for dim in input_shape], name="observation", dtype="uint8")
            self.next_ob_placeholder = Input(shape=[dim for dim in input_shape], name="next_ob", dtype="uint8")
            ob_range = self.env.observation_space.high - self.env.observation_space.low
            cast_layer = Lambda(lambda x: tf.cast(x, tf.float32) / ob_range, name="cast_to_float")
            self.ob_placeholder_float = cast_layer(self.ob_placeholder)
            self.next_ob_placeholder_float = cast_layer(self.next_ob_placeholder)
        else:
            self.ob_placeholder = Input(shape=[dim for dim in input_shape], name="observation", dtype="float32")
            self.next_ob_placeholder = Input(shape=[dim for dim in input_shape], name="next_ob", dtype="float32")
            self.ob_placeholder_float = self.ob_placeholder
            self.next_ob_placeholder_float = self.next_ob_placeholder
        self.ac_placeholder = tf.placeholder(shape=[None], name="action", dtype=tf.int32)
        self.rew_placeholder = tf.placeholder(shape=[None], name="reward", dtype=tf.float32)
        self.done_mask_ph = tf.placeholder(shape=[None], name="done_mask", dtype=tf.float32)
        self.learning_rate = tf.placeholder(shape=(), name="learning_rate", dtype=tf.float32)

    def setup_inference(self):
        # q_op and target_q_op are the Q and target network models
        self.q_func_ob = self.q_model_fn(self.ob_placeholder_float)
        self.q_model = Model(self.ob_placeholder, self.q_func_ob)
        self.q_model.summary()
        # handle for use in rolling out policy
        self.q_func_next_ob = self.q_model_fn(self.next_ob_placeholder_float)

        self.target_q_func_next_ob = self.target_model_fn(self.next_ob_placeholder_float)
        self.target_model = Model(self.next_ob_placeholder, self.target_q_func_next_ob)
        self.target_q_func_vars = self.target_model.trainable_weights

    def setup_loss(self):
        # Define the loss tf operation
        if self.double_q:  # double DQN, use current network to evaluate actions, max_value_action = argmax_{a'} Q_{phi}(s', a')
            max_value_action = tf.argmax(self.q_func_next_ob, axis=1, name="max_value_action", output_type=tf.int32)
        else:  # vanilla DQN, use target network to evaluate actions, max_value_action = argmax_{a'} Q_{phi'}(s', a')
            max_value_action = tf.argmax(self.target_q_func_next_ob, axis=1, name="max_value_action",
                                         output_type=tf.int32)

        # indexing the max valued actions into the target Q values
        indices = tf.stack([tf.range(tf.shape(max_value_action)[0]), max_value_action], axis=1)
        max_Q_value = tf.gather_nd(self.target_q_func_next_ob, indices, name="max_Q_value")

        # y = r + (1 - done) * gamma * Q_{phi'}(s', max_value_action)
        target_values = self.rew_placeholder + (1 - self.done_mask_ph) * self.gamma * max_Q_value

        # indexing the actions into the Q values
        indices = tf.stack([tf.range(tf.shape(self.q_func_ob)[0]), self.ac_placeholder], axis=1)
        action_taken_Q_vals = tf.gather_nd(self.q_func_ob, indices, name="action_Q_value")

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
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.frame_history_len, self.integer_observations)
        self.replay_buffer_idx = None

    def setup_graph(self):
        # we init the model class to setup the model. It can then be called as a function on an input placeholder to
        # return the outputs
        self.q_model_fn = self.q_model_class(self.ac_dim)
        self.target_model_fn = self.target_model_class(self.ac_dim)

        self.setup_placeholders()
        self.setup_inference()
        self.setup_loss()
        self.setup_replay_buffer()
        self.init_tf()

    def init_tf(self):
        # to change tf Session config, see utils.set_keras_session()
        self.tf_sess = keras_backend.get_session()
        self.tf_sess.run(tf.global_variables_initializer())

    def step_env(self):
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
        if self.render_flag:
            self.env.render()
            time.sleep(0.1)

        # step env
        obs, reward, done, info = self.env.step(action)

        if done:
            self.render_flag = False  # only render one episode
            obs = self.env.reset()

        # store this step
        self.replay_buffer.store_effect(idx, action, reward, done)
        self.last_obs = obs

    def update_model(self):
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
        Returns timestep, episode rewards, episode lengths and exploration parameter on logging steps
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
        fpath = os.path.join(self.experiments_path, "models", "model-{}.h5".format(self.t))
        self.q_model.save(filepath=fpath)

    def load_model(self, fpath):
        self.q_model = load_model(fpath, custom_objects={'tf': tf})

    def run_model(self, n_episodes=3, sleep=0.01):
        self.setup_replay_buffer()
        for i in range(n_episodes):

            done = False
            obs = self.env.reset()
            rwd = 0
            while not done:
                # store the latest observation
                idx = self.replay_buffer.store_frame(obs)

                # encode the current observation, [None] adds a dimension of 1 for the batch
                encoded_observation = self.replay_buffer.encode_recent_observation()[None]
                Q_values = self.q_model.predict(encoded_observation)
                action = np.argmax(Q_values)

                # step env
                obs, reward, done, info = self.env.step(action)
                self.env.render()

                rwd += reward

                # store this step
                self.replay_buffer.store_effect(idx, action, reward, done)

                time.sleep(sleep)

            print("Reward: {}".format(rwd))
