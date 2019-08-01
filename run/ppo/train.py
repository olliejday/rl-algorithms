import gym
import os
import argparse
import time
import numpy as np
import roboschool
from mpi4py import MPI
import tensorflow as tf

from src.ppo.ppo import ProximalPolicyOptimisation
from src.ppo.models import FC_NN
from src.ppo.utils import PPOBuffer
from src.common.utils import set_global_seeds, TrainingLogger


def train(env_name, exp_name, seed, debug=True, n_iter=100, save_every=25, **kwargs):
    """
    MPI training function

    :param env_name: Environment name to train on
    :param exp_name: Experiment name to save logs to
    :param seed: Seed to run for this experiment
    :param debug: Debug flag for training (whether to be more reproducible at expense of computation)
    :param n_iter: number of iterations to train for
    :param save_every: number of iterations to save models at
    :param kwargs: arguments to pass to VPG model __init__
    """
    env = gym.make(env_name)
    set_global_seeds(seed, debug)
    env.seed(seed)

    # Setup for MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_processes = comm.Get_size()
    controller = 0
    # Enable gpu usage for just the main process for training
    if rank == controller:
        device_config = tf.ConfigProto()
        root_dir = os.path.dirname(os.path.realpath(__file__))
        experiments_path = os.path.join(root_dir, "experiments", exp_name, str(seed))
        assert not os.path.exists(experiments_path), \
            "Experiment dir {} already exists! Delete it first or use a different dir".format(experiments_path)
    else:
        device_config = tf.ConfigProto(device_count={'GPU': 0})
        experiments_path = None

    ppo = ProximalPolicyOptimisation(env,
                                     comm,
                                     controller,
                                     rank,
                                     experiments_path=experiments_path,
                                     **kwargs)

    if rank == controller:
        log_cols = ["Iteration", "StdReturn", "MaxReturn", "MinReturn", "EpLenMean", "EpLenStd", "Entropy", "KL"]
        training_logger = TrainingLogger(experiments_path, log_cols, config=[str(ppo)])

    ppo.setup_graph()

    timesteps = 0

    for itr in range(1, n_iter + 1):
        # gather data
        buffer = ppo.sample_trajectories(itr)
        # get one long sequence of observations and actions
        # we keep rewards in trajectories until we discount
        obs = buffer.get_obs()
        acs = buffer.get_acs()
        logprobs = buffer.get_logprobs()
        # rewards kept within episodes, undiscounted for logging
        raw_rwds = buffer.rwds
        # advantages and rewards to go with GAE for updates
        rwds, advs = buffer.get_gae(ppo.gamma, ppo.gae_lambda)

        approx_entropy, approx_kl = ppo.update_parameters(obs, acs, rwds, advs, logprobs)

        ppo.sync_weights()

        returns = [sum(r) for r in raw_rwds]
        ep_lens = [len(r) for r in raw_rwds]
        timesteps += np.sum(ep_lens)

        # only log and save model in the controller
        if rank == controller:
            # TODO: need to combine logs over procs
            training_logger.log(Time=time.strftime("%d/%m/%Y %H:%M:%S"),
                                MeanReturn=np.mean(returns),
                                Timesteps=timesteps,
                                Iteration=itr,
                                StdReturn=np.std(returns),
                                MaxReturn=np.max(returns),
                                MinReturn=np.min(returns),
                                EpLenMean=np.mean(ep_lens),
                                EpLenStd=np.std(ep_lens),
                                Entropy=approx_entropy,
                                KL=approx_kl
                                )

            if itr % save_every == 0:
                ppo.save_model(timesteps)

    env.close()


def train_cartpole(n_experiments=3, seed=1, debug=True, exp_name="ppo-cartpole"):
    value_fn = FC_NN([64, 64], 1)
    for i in range(1, n_experiments + 1):
        seed += 10 * i
        train("CartPole-v1", exp_name, seed=seed, debug=debug, value_fn=value_fn,
              min_timesteps_per_batch=2500, n_iter=25, render_every=1000)


def train_inverted_pendulum(n_experiments=3, seed=1, debug=True, exp_name="ppo-inverted-pendulum"):
    value_fn = FC_NN([64, 64], 1)
    for i in range(1, n_experiments + 1):
        seed += 10 * i
        train("RoboschoolInvertedPendulum-v1", exp_name, seed=seed, debug=debug,
              value_fn=value_fn, min_timesteps_per_batch=5000, n_iter=50, render_every=1000, save_every=45)


def train_lander(n_experiments=3, seed=123, debug=False, exp_name="ppo-lander"):
    value_fn = FC_NN([64, 64], 1)
    for i in range(1, n_experiments + 1):
        seed += 10 * i
        train("LunarLanderContinuous-v2", exp_name, seed=seed, debug=debug, value_fn=value_fn,
              min_timesteps_per_batch=40000, render_every=1000, save_every=90)


def train_half_cheetah(n_experiments=3, seed=1, debug=False, exp_name="ppo-half-cheetah"):
    value_fn = FC_NN([64, 64], 1)
    for i in range(1, n_experiments + 1):
        seed += 10 * i
        train("RoboschoolHalfCheetah-v1", exp_name, n_experiments, seed=seed, debug=debug, value_fn=value_fn,
              min_timesteps_per_batch=50000, render_every=1000, save_every=90)


if __name__ == "__main__":
    options = {}
    options['lander'] = train_lander
    options['inverted-pendulum'] = train_inverted_pendulum
    options['cartpole'] = train_cartpole
    options['half-cheetah'] = train_half_cheetah

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=options.keys())

    args = parser.parse_args()

    options[args.experiment]()
