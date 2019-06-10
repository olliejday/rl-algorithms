import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import time

import src.sac.models as nn
from src.common.utils import set_global_seeds, TrainingLogger
from src.sac.sac import SAC
import src.sac.utils as utils

from multiprocessing import Process


def train(env_name, exp_name, n_experiments=3, seed=1, debug=True):
    """
    General training setup. Just an interface to call _train() for each seed in parallel.
    :param env_name: Environment name to train on
    :param exp_name: Experiment name to save logs to
    :param n_experiments: number of seeds of experiments to run
    :param seed: seed for first experiment
    :param debug: debug flag for training setup and seeding
    """

    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_dir = os.path.join(root_dir, "experiments", exp_name)
    assert not os.path.exists(experiments_dir), \
        "Experiment dir {} already exists! Delete it first or use a different dir".format(experiments_dir)

    processes = []

    for e in range(n_experiments):
        seed += 10 * e
        print('Running experiment with seed %d'%seed)

        def train_func():
            _train(
                env_name=env_name,
                exp_name=exp_name,
                seed=seed,
                debug=debug
            )
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


def _train(env_name, exp_name, seed, debug=True):
    """
     Training function to be called for a process in parallel, same args as train()
    """
    alpha = {
        'Ant-v2': 0.1,
        'HalfCheetah-v2': 0.2,
        'Hopper-v2': 0.2,
        'Humanoid-v2': 0.05,
        'Walker2d-v2': 0.2,
    }.get(env_name, 0.2)

    algorithm_params = {
        'alpha': alpha,
        'batch_size': 256,
        'discount': 0.99,
        'learning_rate': 3e-4,
        'reparameterize': True,
        'tau': 5e-3,
        'epoch_length': 1000,
        'n_epochs': 500,
        'two_qf': True,
    }
    sampler_params = {
        'max_episode_length': 1000,
        'prefill_steps': 1000,
    }
    replay_pool_params = {
        'max_size': 1e6,
    }

    value_function_params = {
        'hidden_layer_sizes': (128, 128),
    }

    q_function_params = {
        'hidden_layer_sizes': (128, 128),
    }

    policy_params = {
        'hidden_layer_sizes': (128, 128),
    }

    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_path = os.path.join(root_dir, "experiments", exp_name, str(seed))

    params = {
        'exp_name': exp_name,
        'env_name': env_name,
        'algorithm_params': algorithm_params,
        'sampler_params': sampler_params,
        'replay_pool_params': replay_pool_params,
        'value_function_params': value_function_params,
        'q_function_params': q_function_params,
        'policy_params': policy_params
    }
    log_cols = ["Iteration", "StdReturn", "MaxReturn", "MinReturn", "EpLenMean", "EpLenStd", "NEpisodes"]
    training_logger = TrainingLogger(experiments_path, log_cols, config=[str(params)])

    # Set random seeds
    env = gym.make(env_name)
    set_global_seeds(seed, debug)
    env.seed(seed)

    sampler = utils.SimpleSampler(**sampler_params)
    replay_pool = utils.SimpleReplayPool(
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        **replay_pool_params)

    q_function = nn.QFunction(name='q_function', **q_function_params)
    if algorithm_params.get('two_qf', False):
        q_function2 = nn.QFunction(name='q_function2', **q_function_params)
    else:
        q_function2 = None
    value_function = nn.ValueFunction(
        name='value_function', **value_function_params)
    target_value_function = nn.ValueFunction(
        name='target_value_function', **value_function_params)
    policy = nn.GaussianPolicy(
        action_dim=env.action_space.shape[0],
        reparameterize=algorithm_params['reparameterize'],
        **policy_params)

    sampler.initialize(env, policy, replay_pool)

    algorithm = SAC(**algorithm_params)

    algorithm.build(
        env=env,
        policy=policy,
        q_function=q_function,
        q_function2=q_function2,
        value_function=value_function,
        target_value_function=target_value_function)

    for epoch in algorithm.train(sampler, n_epochs=algorithm_params.get('n_epochs', 1000)):
        ep_rtns, ep_lens, timesteps, n_eps = sampler.get_statistics()
        training_logger.log(Time=time.strftime("%d/%m/%Y %H:%M:%S"),
                            MeanReturn=np.mean(ep_rtns),
                            Timesteps=timesteps,
                            Iteration=epoch,
                            StdReturn=np.std(ep_rtns),
                            MaxReturn=np.max(ep_rtns),
                            MinReturn=np.min(ep_rtns),
                            EpLenMean=np.mean(ep_lens),
                            EpLenStd=np.std(ep_lens),
                            NEpisodes=n_eps,
                            )


#TODO
def train_lander():
    train('LunarLanderContinuous-v2', "sac-lander", seed=1, debug=True)

#TODO
def train_half_cheetah():
    train("RoboschoolHalfCheetah-v1", "sac-half-cheetah", seed=1, debug=True)

#TODO
def train_inverted_pendulum():
    train("RoboschoolInvertedPendulum-v1", "sac-inverted-pendulum", seed=1, debug=True)

#TODO
def train_ant():
    train("RoboschoolAnt-v1", "sac-ant", seed=1, debug=True)


if __name__ == "__main__":
    options = {}
    options['lander'] = train_lander
    options['half-cheetah'] = train_half_cheetah
    options['inverted-pendulum'] = train_inverted_pendulum
    options['ant'] = train_ant

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=options.keys())

    args = parser.parse_args()

    options[args.experiment]()
