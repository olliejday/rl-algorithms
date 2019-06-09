import time
from multiprocessing import Process
import gym
import os
import argparse
import numpy as np
import roboschool

from src.ac.ac import ActorCrtic
from src.common.utils import set_global_seeds, TrainingLogger
from src.common.models import FC_NN


def train(env_name, exp_name, n_experiments=3, seed=123, debug=True, n_iter=100, save_every=25, **kwargs):
    """
        General training setup, just an interface to call _train() for each seed in parallel.

    :param env_name: Environment to train on
    :param exp_name: Experiment name to save logs to
    :param n_experiments: number of different seeds to run
    :param seed: Seed to run this experiment with
    :param debug: Debug flag for training (whether to be more reproducible at expense of computation)
    :param n_iter: number of iterations to train for
    :param save_every: number of iterations to save models at
    :param kwargs: arguments to pass to AC model __init__
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

    ac = ActorCrtic(env,
                    experiments_path=experiments_path,
                    **kwargs)

    log_cols = ["Iteration", "StdReturn", "MaxReturn", "MinReturn", "EpLenMean", "EpLenStd", "Entropy", "KL"]
    training_logger = TrainingLogger(experiments_path, log_cols, config=[str(ac)])

    ac.setup_graph()

    timesteps = 0

    for itr in range(1, n_iter + 1):
        buffer = ac.sample_trajectories(itr)

        obs, acs, rwds, next_obs, terminals, logprobs = buffer.get()

        ac.update_parameters(obs, next_obs, rwds, terminals, acs)
        approx_entropy, approx_kl = ac.training_metrics(obs, acs, logprobs)

        # use buffer.rwds to keep rewards in their lists of episodes
        returns = [sum(r) for r in buffer.rwds]
        ep_lens = [len(r) for r in buffer.rwds]
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
            ac.save_model(timesteps)

    env.close()


def train_cartpole(seed=123, debug=True, exp_name="ac-cartpole"):
    train("CartPole-v1", exp_name, seed=seed, debug=debug, min_timesteps_per_batch=2000,
          n_iter=30, render_every=1000, num_target_updates=10,
          num_grad_steps_per_target_update=10)


def train_inverted_pendulum(seed=123, debug=True, exp_name="ac-inverted-pendulum"):
    train("RoboschoolInvertedPendulum-v1", exp_name, seed=seed, debug=debug, min_timesteps_per_batch=5000,
          discrete=False, learning_rate_actor=0.01, learning_rate_critic=0.01, n_iter=30, gamma=0.95,
          render_every=1000, save_every=30)


def train_half_cheetah(seed=123, debug=False, exp_name="ac-half-cheetah"):
    train("RoboschoolHalfCheetah-v1", exp_name, seed=seed, debug=debug, discrete=False, min_timesteps_per_batch=30000,
          render_every=1000, gamma=0.95, learning_rate_actor=0.01, learning_rate_critic=0.01,
          critic_model_class=FC_NN, hidden_layer_sizes=[64, 64])


if __name__ == "__main__":
    options = {}
    options['inverted-pendulum'] = train_inverted_pendulum
    options['cartpole'] = train_cartpole
    options['half-cheetah'] = train_half_cheetah

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=options.keys())

    args = parser.parse_args()

    options[args.experiment]()
