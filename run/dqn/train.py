import gym
import os
import tensorflow as tf
import argparse

from src.dqn.models import DQNFCModelKeras, DQNCNNModelKerasSmall
from src.dqn.dqn import DQN
from src.dqn.utils import DQNTrainingLogger, get_env, PiecewiseSchedule, ConstantSchedule, OptimizerSpec
from src.dqn.atari_wrappers import wrap_deepmind


def train(exp_name,
          env,
          model_class,
          optimizer_spec,
          num_timesteps=1e8,
          **kwargs):
    """
    General training setup.
    :param env: Environment to train on
    :param exp_name: Experiment name to save logs to
    :param model_class: Model class to call to generate the DQN (see src.dqn.models)
    :param optimizer_spec: holds an optimzer and it's learning rate schedule (see src.dqn.utils)
    :param num_timesteps: number of timesteps to train for
    :param kwargs: arguments to pass to DQN model __init__
    """
    # setup logging dirs
    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_path = os.path.join(root_dir, "experiments", exp_name)

    dqn = DQN(env=env,
              model_class=model_class,
              optimizer_spec=optimizer_spec,
              experiments_path=experiments_path,
              **kwargs)
    dqn.setup_graph()

    training_logger = DQNTrainingLogger(experiments_path, ["Model_fn: {}".format(model_class.__name__),
                                                            str(dqn)])

    while dqn.t < num_timesteps:
        dqn.step_env()
        dqn.update_model()
        logs = dqn.log_progress()
        if logs is not None:
            timestep, ep_rtrns, ep_lens, exploration = logs
            training_logger.log(timestep, ep_rtrns, ep_lens, exploration)

    # save final model
    dqn.save_model()

    env.close()


def train_lander(seed=123, debug=True):
    """
    :param seed: seed to setup system
    :param debug: debug flag for seeding reproducibility vs performance
    """
    num_timesteps = 5e5

    exploration = PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )

    lr_schedule = PiecewiseSchedule([
        (0, 1e-3),
        (num_timesteps / 2, 5e-4),
        (3 * num_timesteps / 4, 1e-4),
    ],
        outside_value=1e-4)


    optimizer = OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        lr_schedule=lr_schedule,
        kwargs={}
    )

    # setup env and wrap in monitor
    env = gym.make("LunarLander-v2")
    env = get_env(env, seed, debug)

    train("dqn_lander",
          env,
          DQNFCModelKeras,
          optimizer,
          exploration=exploration,
          num_timesteps=num_timesteps,
          replay_buffer_size=50000,
          gamma=1.0,
          learning_starts=1000,
          learning_freq=1,
          frame_history_len=1,
          target_update_freq=3000,
          save_every=1e5,
          batch_size=32,
          grad_norm_clipping=10,
          delta=1.0,
          double_q=True,
          log_every_n_steps=10000,
          integer_observations=False)


def train_pong(seed=123, debug=False):
    """
    :param seed: seed to setup system
    :param debug: debug flag for seeding reproducibility vs performance
    """
    num_timesteps = 1e8
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
        (0, 1e-4 * lr_multiplier),
        (num_iterations / 10, 1e-4 * lr_multiplier),
        (num_iterations / 2, 5e-5 * lr_multiplier),
    ],
        outside_value=5e-5 * lr_multiplier)

    optimizer = OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    exploration = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    # setup env and wrap in monitor
    env = gym.make("PongNoFrameskip-v4")
    env = get_env(env, seed, debug)
    env = wrap_deepmind(env)

    train("dqn-pong",
          env,
          DQNCNNModelKerasSmall,
          optimizer,
          num_timesteps=num_timesteps,
          exploration=exploration,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10,
          delta=1.0,
          save_every=5e5,
          double_q=True,
          log_every_n_steps=10000,
          integer_observations=True)


if __name__ == "__main__":
    options = {}
    options['lander'] = train_lander
    options['pong'] = train_pong

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=options.keys())

    args = parser.parse_args()

    options[args.experiment]()