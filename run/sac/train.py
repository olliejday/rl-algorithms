import argparse
import gym
import numpy as np
import os
import time
import roboschool
from multiprocessing import Process


from src.common.utils import set_global_seeds, TrainingLogger
from src.sac.sac import SAC
import src.sac.utils as utils



def train(env_name, exp_name, algorithm_params, n_experiments=3, seed=1, debug=True, n_epochs=1000, save_every=300):
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
                algorithm_params=algorithm_params,
                debug=debug,
                save_every=save_every,
                n_epochs=n_epochs
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


def _train(env_name, exp_name, seed, algorithm_params, debug=True, save_every=450, n_epochs=500):
    """
     Training function to be called for a process in parallel, same args as train()
    """
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

    sac = SAC(env,
              experiments_dir=experiments_path,
              **algorithm_params)

    sac.build(
        q_function_params=q_function_params,
        value_function_params=value_function_params,
        policy_params=policy_params)

    sampler.initialize(env, sac.policy, replay_pool)

    for epoch in sac.train(sampler, n_epochs=n_epochs):
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
        if epoch % save_every == 0:
            sac.save_model(timesteps)

# TODO: 5. test lander (discrete) - cf. DQN
# TODO: 8. test atari (discrete) - cf. DQN

def train_cartpole():
    # TODO: 4. test cartpole (discrete) - cf. PPO etc.
    algorithm_params = {
        'alpha': 0.2,
        'batch_size': 256,
        'discount': 0.99,
        'learning_rate': 5e-4,
        'tau': 5e-2,
        'epoch_length': 100,
        'two_qf': False
    }
    train('CartPole-v1', "sac-cartpole", algorithm_params, n_epochs=1000, save_every=490, seed=1, debug=True)


def train_lander():
    algorithm_params = {
        'alpha': 0.2,
        'batch_size': 256,
        'discount': 0.99,
        'learning_rate': 3e-4,
        'reparameterize': True,
        'tau': 5e-3,
        'epoch_length': 1000,
        'two_qf': False
    }
    train('LunarLanderContinuous-v2', "sac-lander", algorithm_params, n_epochs=1500, save_every=740,
          seed=123, debug=True)


def train_half_cheetah():
    algorithm_params = {
        'alpha': 0.2,
        'batch_size': 256,
        'discount': 0.99,
        'learning_rate': 5e-4,
        'reparameterize': True,
        'tau': 5e-3,
        'epoch_length': 1000,
        'two_qf': True
    }
    train("RoboschoolHalfCheetah-v1", "sac-half-cheetah", algorithm_params, n_epochs=4000, save_every=950, seed=1, debug=True)


def train_inverted_pendulum():
    algorithm_params = {
        'alpha': 0.2,
        'batch_size': 256,
        'discount': 0.99,
        'learning_rate': 6e-4,
        'reparameterize': True,
        'tau': 5e-3,
        'epoch_length': 500,
        'two_qf': True
    }
    train("RoboschoolInvertedPendulum-v1", "sac-inverted-pendulum", algorithm_params, n_epochs=500, save_every=490,
          seed=1, debug=True)


def train_ant():
    algorithm_params = {
        'alpha': 0.1,
        'batch_size': 256,
        'discount': 0.99,
        'learning_rate': 3e-4,
        'reparameterize': True,
        'tau': 5e-3,
        'epoch_length': 1000,
        'two_qf': True
    }
    train("RoboschoolAnt-v1", "sac-ant", algorithm_params, seed=1, debug=True)


if __name__ == "__main__":
    options = {}
    options['cartpole'] = train_cartpole
    options['lander'] = train_lander
    options['half-cheetah'] = train_half_cheetah
    options['inverted-pendulum'] = train_inverted_pendulum
    options['ant'] = train_ant

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=options.keys())

    args = parser.parse_args()

    options[args.experiment]()
