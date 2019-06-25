import gym
import os
import argparse
import time
import numpy as np
import roboschool
from multiprocessing import Process

from src.ppo.ppo import ProximalPolicyOptimisation
from src.ppo.models import FC_NN
from src.common.utils import set_global_seeds, TrainingLogger


def train(env_name, exp_name, n_experiments, seed=123, debug=True, n_iter=100, save_every=25, **kwargs):
    """
        General training setup, just an interface to call _train() for each seed in parallel.

    :param env_name: Environment name to train on
    :param exp_name: Experiment name to save logs to
    :param n_experiments: number of different seeds to run
    :param seed: Seed to run for this experiment
    :param debug: Debug flag for training (whether to be more reproducible at expense of computation)
    :param n_iter: number of iterations to train for
    :param save_every: number of iterations to save models at
    :param kwargs: arguments to pass to VPG model __init__
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_dir = os.path.join(root_dir, "experiments", exp_name)
    assert not os.path.exists(experiments_dir), \
        "Experiment dir {} already exists! Delete it first or use a different dir".format(experiments_dir)

    processes = []

    for i in range(n_experiments):
        seed += 10 * i

        def train_func():
            _train(env_name, exp_name, seed, debug=debug, n_iter=n_iter, save_every=save_every, **kwargs)

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


def _train(env_name, exp_name, seed, debug=True, n_iter=100, save_every=25, **kwargs):
    """
    Training function to be called for a process in parallel, same args as train()
    """

    env = gym.make(env_name)
    set_global_seeds(seed, debug)
    env.seed(seed)

    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_path = os.path.join(root_dir, "experiments", exp_name, str(seed))

    ppo = ProximalPolicyOptimisation(env,
                                 experiments_path=experiments_path,
                                 **kwargs)

    log_cols = ["Iteration", "StdReturn", "MaxReturn", "MinReturn", "EpLenMean", "EpLenStd", "Entropy", "KL"]
    training_logger = TrainingLogger(experiments_path, log_cols, config=[str(ppo)])

    ppo.setup_graph()

    timesteps = 0

    for itr in range(1, n_iter + 1):
        buffer = ppo.sample_trajectories(itr)

        # get one long sequence of observations and actions
        # we keep rewards in trajectories until we discount
        obs = buffer.get_obs()
        acs = buffer.get_acs()
        logprobs = buffer.get_logprobs()
        rwds = buffer.rwds

        q_n, adv_n = ppo.estimate_return(obs, rwds)
        approx_entropy, approx_kl = ppo.update_parameters(obs, acs, q_n, adv_n, logprobs)

        returns = [sum(r) for r in rwds]
        ep_lens = [len(r) for r in rwds]
        timesteps += np.sum(ep_lens)

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

# TODO: train all these -- on longjob?? on colab??, plot, plot comparisons


def train_cartpole(n_experiments=3, seed=1, debug=True, exp_name="ppo-cartpole"):
    nn_baseline = FC_NN([64, 64], 1)
    train("CartPole-v1", exp_name, n_experiments, seed=seed, debug=debug, nn_baseline=nn_baseline,
          min_timesteps_per_batch=2500, learning_rate=0.02, n_iter=25, render_every=1000)


def train_inverted_pendulum(n_experiments=3, seed=1, debug=True, exp_name="ppo-inverted-pendulum"):
    nn_baseline = FC_NN([64, 64], 1)
    train("RoboschoolInvertedPendulum-v1", exp_name, n_experiments, seed=seed, debug=debug,
          nn_baseline=nn_baseline, min_timesteps_per_batch=5000,
          learning_rate=0.005, n_iter=50, gamma=0.95, render_every=1000, save_every=45)


def train_lander(n_experiments=3, seed=123, debug=False, exp_name="ppo-lander"):
    nn_baseline = FC_NN([64, 64], 1)
    train("LunarLanderContinuous-v2", exp_name, n_experiments, seed=seed, debug=debug, nn_baseline=nn_baseline,
          min_timesteps_per_batch=40000, learning_rate=0.005, gradient_batch_size=40000,
          render_every=1000, save_every=90)


def train_half_cheetah(n_experiments=3, seed=1, debug=False, exp_name="ppo-half-cheetah"):
    nn_baseline = FC_NN([64, 64], 1)
    train("RoboschoolHalfCheetah-v1", exp_name, n_experiments, seed=seed, debug=debug, nn_baseline=nn_baseline,
          min_timesteps_per_batch=50000, learning_rate=0.01, gradient_batch_size=50000,
          render_every=1000, save_every=90)


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
