import gym
import os
import tensorflow as tf

from src.dqn.models import DQNFCModelKeras, DQNCNNModelKerasSmall
from src.dqn.dqn import DQN
from src.dqn.utils import DQNTrainingLogger, get_env, PiecewiseSchedule, ConstantSchedule, OptimizerSpec


def train(exp_name,
          env_name,
          model_class,
          optimizer_spec,
          seed=123,
          num_timesteps=1e8,
          debug=False,
          **kwargs):
    """
    General training setup.
    :param env_name: Environment name to train on
    :param exp_name: Experiment name to save logs to
    :param model_class: Model class to call to generate the DQN (see src.dqn.models)
    :param optimizer_spec: holds an optimzer and it's learning rate schedule (see src.dqn.utils)
    :param seed: seed to setup system
    :param num_timesteps: number of timesteps to train for
    :param debug: debug flag for seeding reproducibility vs performance
    :param kwargs: arguments to pass to DQN model __init__
    """

    # setup env and wrap in monitor
    env = gym.make(env_name)
    env = get_env(env, seed, debug)

    # setup logging dirs
    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_path = os.path.join(root_dir, "experiments", exp_name)

    dqn = DQN(env=env,
              model_class=model_class,
              optimizer_spec=optimizer_spec,
              experiments_path=experiments_path,
              **kwargs)
    dqn.setup_graph()

    training_logger = DQNTrainingLogger(experiments_path, ["Env_name: {}".format(env_name),
                                                        "Model_fn: {}".format(model_class.__name__),
                                                        "Seed: {}".format(seed),
                                                           str(kwargs),
                                                           str(dqn)])

    for _ in range(int(num_timesteps)):
        dqn.step_env()
        dqn.update_model()
        logs = dqn.log_progress()
        if logs is not None:
            timestep, ep_rtrns, ep_lens, exploration = logs
            training_logger.log(timestep, ep_rtrns, ep_lens, exploration)

    # save final model
    dqn.save_model()

    env.close()


def train_lander():
    num_timesteps = 5e5

    exploration = PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )

    lr_schedule = ConstantSchedule(1e-3)

    optimizer = OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        lr_schedule=lr_schedule,
        kwargs={}
    )
    train("dqn_lander1",
          "LunarLander-v2",
          DQNFCModelKeras,
          optimizer,
          debug=True,
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


def train_snake(env_type):
    assert env_type == "stacked" or env_type == "grid"

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

    train("dqn_snake_{}".format(env_type),
          "snake-{}-v0".format(env_type),
          DQNCNNModelKerasSmall,
          optimizer,
          num_timesteps=num_timesteps,
          exploration=exploration,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=5e4,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10,
          delta=1.0,
          save_every=2e6,
          double_q=True,
          log_every_n_steps=10000,
          integer_observations=True)


if __name__ == "__main__":
    train_lander()
    # train_snake("grid")
