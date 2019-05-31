import gym
import os
import argparse

from src.vpg.vpg import run_model
from src.vpg.models import fc_small, fc_medium
from src.common.utils import set_global_seeds


def run(env_name,
        exp_name,
        model_fn,
        model_path=None,
        seed=123,
        debug=False,
        **kwargs):
    """
    General running setup for saved models.
    :param exp_name: experiment directory to look for models in
    :param env_name: environment to run model in
    :param model_number: model number to load, if None then latest model is loaded
    :param seed: seed to set for system
    :param debug: debug flag for seeding reproducibility vs performance
    """
    env = gym.make(env_name)

    # Set random seeds
    set_global_seeds(seed, debug)

    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_dir = os.path.join(root_dir, "experiments", exp_name)
    models_dir = os.path.join(experiments_dir, "models")

    assert os.path.exists(models_dir), "Invalid experiment name, models directory does not exist for" \
                                       ": {}, at: {}".format(exp_name, models_dir)

    run_model(env, model_fn, experiments_dir, model_path=model_path, n_episodes=3, **kwargs)

    env.close()


def run_lander(exp_name="vpg-lander"):
    run("LunarLanderContinuous-v2", exp_name, fc_small, debug=True, discrete=False)


def run_cartpole(exp_name="vpg-cartpole"):
    run("CartPole-v0", exp_name, fc_small, debug=True)


if __name__ == "__main__":
    options = {}
    options['lander'] = run_lander
    options['cartpole'] = run_cartpole

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=options.keys())

    args = parser.parse_args()

    options[args.experiment]()