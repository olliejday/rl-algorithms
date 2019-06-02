import gym
import os
import argparse
import roboschool

from src.ac.ac import run_model
from src.ac.models import fc
from src.common.utils import set_global_seeds


def run(env,
        exp_name,
        model_fn,
        model_path=None,
        **kwargs):
    """
    General running setup for saved models.
    :param env: environment to run model in
    :param exp_name: experiment directory to look for models in
    :param model_fn: model function to load for policy
    :param model_path: model path to load, if None then latest model  in exp_name directory is loaded
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_dir = os.path.join(root_dir, "experiments", exp_name)
    models_dir = os.path.join(experiments_dir, "models")

    assert os.path.exists(models_dir), "Invalid experiment name, models directory does not exist for" \
                                       ": {}, at: {}".format(exp_name, models_dir)

    run_model(env, model_fn, experiments_dir, model_path=model_path, n_episodes=3, **kwargs)

    env.close()


def run_cartpole(exp_name="ac-cartpole", seed=123, debug=True):
    env = gym.make("CartPole-v0")
    set_global_seeds(seed, debug)
    env.seed(seed)
    run(env, exp_name, fc, discrete=True)


def run_inverted_pendulum(exp_name="ac-inverted-pendulum", seed=123, debug=True):
    env = gym.make("RoboschoolInvertedPendulum-v1")
    set_global_seeds(seed, debug)
    env.seed(seed)
    run(env, exp_name, fc, discrete=False)


def run_half_cheetah(exp_name="ac-half-cheetah", seed=123, debug=False):
    env = gym.make("RoboschoolHalfCheetah-v1")
    set_global_seeds(seed, debug)
    env.seed(seed)
    run(env, exp_name, fc, discrete=False)

if __name__ == "__main__":
    options = {}
    options['inverted-pendulum'] = run_inverted_pendulum
    options['cartpole'] = run_cartpole
    options['half-cheetah'] = run_half_cheetah


    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=options.keys())

    args = parser.parse_args()

    options[args.experiment]()