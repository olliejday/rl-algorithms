import gym
import os
import argparse
import roboschool

from src.vpg.vpg import run_model
from src.common.utils import set_global_seeds


def run(env,
        exp_name,
        seed=123,
        model_path=None,
        **kwargs):
    """
    General running setup for saved models.
    :param env: environment to run model in
    :param exp_name: experiment directory to look for models in
    :param model_path: model path to load, if None then latest model  in exp_name directory is loaded
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_dir = os.path.join(root_dir, "experiments", exp_name, str(seed))
    models_dir = os.path.join(experiments_dir, "models")

    assert os.path.exists(models_dir), "Invalid experiment name, models directory does not exist for" \
                                       ": {}, {} at: {}".format(exp_name, str(seed), models_dir)

    run_model(env, experiments_dir, model_path=model_path, n_episodes=3, **kwargs)

    env.close()


def run_cartpole(exp_name="vpg-cartpole", seed=123, debug=True):
    env = gym.make("CartPole-v1")
    set_global_seeds(seed, debug)
    env.seed(seed)
    run(env, exp_name, seed, discrete=True)


def run_inverted_pendulum(exp_name="vpg-inverted-pendulum", seed=11, debug=True):
    env = gym.make("RoboschoolInvertedPendulum-v1")
    set_global_seeds(seed, debug)
    env.seed(seed)
    run(env, exp_name, seed, discrete=False)


def run_lander(exp_name="vpg-lander", seed=123, debug=True):
    env = gym.make("LunarLanderContinuous-v2")
    set_global_seeds(seed, debug)
    env.seed(seed)
    run(env, exp_name, seed, discrete=False)


def run_half_cheetah(exp_name="vpg-half-cheetah", seed=123, debug=False):
    env = gym.make("RoboschoolHalfCheetah-v1")
    set_global_seeds(seed, debug)
    env.seed(seed)
    run(env, exp_name, seed, discrete=False)


if __name__ == "__main__":
    options = {}
    options['lander'] = run_lander
    options['inverted-pendulum'] = run_inverted_pendulum
    options['cartpole'] = run_cartpole
    options['half-cheetah'] = run_half_cheetah


    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=options.keys())

    args = parser.parse_args()

    options[args.experiment]()