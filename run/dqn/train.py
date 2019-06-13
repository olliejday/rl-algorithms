import gym
import os
import tensorflow as tf
import argparse
import time
import numpy as np
from  multiprocessing import Process

from src.dqn.models import DQNFCModelKeras, DQNCNNModelKerasSmall
from src.dqn.dqn import DQN
from src.dqn.utils import get_env, PiecewiseSchedule, ConstantSchedule, OptimizerSpec
from src.dqn.atari_wrappers import wrap_deepmind

from src.common.utils import TrainingLogger


def train(exp_name,
          env_name,
          model_class,
          optimizer_spec,
          n_experiments=3,
          wrapper=None,
          seed=123,
          debug=True,
          num_timesteps=1e8,
          mean_n=100,
          **kwargs):
    """
    General training setup. Just an interface to call _train() for each seed in parallel.
    :param env_name: Environment name to train on
    :param exp_name: Experiment name to save logs to
    :param model_class: Model class to call to generate the DQN (see src.dqn.models)
    :param optimizer_spec: holds an optimzer and it's learning rate schedule (see src.dqn.utils)
    :param wrapper: default None, wrapper to wrap env in
    :param seed: seed to seed env and system
    :param debug: debug flag for training
    :param num_timesteps: number of timesteps to train for
    :param mean_n: number of episodes to average over for logging
    :param kwargs: arguments to pass to DQN model __init__
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_dir = os.path.join(root_dir, "experiments", exp_name)
    assert not os.path.exists(experiments_dir), \
        "Experiment dir {} already exists! Delete it first or use a different dir".format(experiments_dir)

    processes = []

    for i in range(n_experiments):
        seed += 10 * i

        def train_func():
            _train(exp_name, env_name, model_class, optimizer_spec, wrapper=wrapper, seed=seed, debug=debug,
                   num_timesteps=num_timesteps, mean_n=mean_n, **kwargs)

        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block
        # until this process finishes
        # p.join()

    for p in processes:
        p.join()


def _train(exp_name,
          env_name,
          model_class,
          optimizer_spec,
          wrapper=None,
          seed=123,
          debug=True,
          num_timesteps=1e8,
          mean_n=100,
          **kwargs):
    """
    Training function to be called for a process in parallel, same args as train()
    """
    # setup env and wrap in monitor
    env = gym.make(env_name)
    env = get_env(env, seed, debug)
    if wrapper is not None:
        env = wrapper(env)

    # setup logging dirs
    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_dir = os.path.join(root_dir, "experiments", exp_name, str(seed))

    dqn = DQN(env=env,
              model_class=model_class,
              optimizer_spec=optimizer_spec,
              experiments_dir=experiments_dir,
              **kwargs)
    dqn.setup_graph()

    log_cols = ["StdReturn", "MaxReturn", "MinReturn", "EpLenMean", "EpLenStd", "Exploration"]
    training_logger = TrainingLogger(experiments_dir, log_cols,
                                     config=["Model_fn: {}".format(model_class.__name__), str(dqn)])

    while dqn.t < num_timesteps:
        dqn.step_env()
        dqn.update_model()
        logs = dqn.log_progress()
        if logs is not None:
            timesteps, ep_rtrns, ep_lens, exploration = logs

            if len(ep_rtrns) > mean_n:
                # last 100 episdoes
                ep_rtrns = ep_rtrns[-mean_n:]
                ep_lens = ep_lens[-mean_n:]
            training_logger.log(Time=time.strftime("%d/%m/%Y %H:%M:%S"),
                                MeanReturn=np.mean(ep_rtrns),
                                Timesteps=timesteps,
                                StdReturn=np.std(ep_rtrns),
                                MaxReturn=np.max(ep_rtrns),
                                MinReturn=np.min(ep_rtrns),
                                EpLenMean=np.mean(ep_lens),
                                EpLenStd=np.std(ep_lens),
                                Exploration=exploration,
                                )

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

    train("dqn-lander",
          "LunarLander-v2",
          DQNFCModelKeras,
          optimizer,
          seed=seed,
          debug=debug,
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


def train_pong(seed=1, debug=False):
    """
    :param seed: seed to setup system
    :param debug: debug flag for seeding reproducibility vs performance
    """
    num_timesteps = 5e6

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
        (0, 1e-4 * lr_multiplier),
        (num_timesteps / 2, 1e-4 * lr_multiplier),
        (num_timesteps, 8.5e-5 * lr_multiplier),
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
            (num_timesteps / 5, 0.1),
            (num_timesteps, 0.075),
        ], outside_value=0.01
    )

    train("dqn-pong",
          "PongNoFrameskip-v4",
          DQNCNNModelKerasSmall,
          optimizer,
          wrapper=wrap_deepmind,
          n_experiments=1,
          seed=seed,
          debug=debug,
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
          save_every=1e6,
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