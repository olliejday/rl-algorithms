import gym
import os
import argparse

from src.dqn.dqn import run_model
from src.dqn.utils import get_env


def run(exp_name,
        env_name,
        frame_history_len,
        integer_observations,
        model_number=None,
        seed=123,
        debug=False):
    """
    General running setup for saved models.
    :param exp_name: experiment directory to look for models in
    :param env_name: environment to run model in
    :param frame_history_len: how many observations to stack in a sequence to pass to model
    :param integer_observations: whether to store integer observations in replay buffer
    :param model_number: model number to load, if None then latest model is loaded
    :param seed: seed to set for system
    :param debug: debug flag for seeding reproducibility vs performance
    """
    print('Random seed = %d' % seed)
    env = gym.make(env_name)
    env = get_env(env, seed, debug)

    root_dir = os.path.dirname(os.path.realpath(__file__))
    models_dir = os.path.join(root_dir, "experiments", exp_name, "models")

    assert os.path.exists(models_dir), "Invalid experiment name, models directory does not exist for" \
                                       ": {}, at: {}".format(exp_name, models_dir)

    if model_number is None:
        # then get latest model
        model_files = os.listdir(models_dir)
        model_number = max([int(f.split(".")[0].split("-")[1]) for f in model_files])
    model_path = os.path.join(models_dir, "model-{}.h5".format(model_number))

    assert os.path.exists(model_path), "Invalid model number, models file does not exist for" \
                                       ": {}, at: {}".format(model_number, model_path)

    run_model(env, model_path, frame_history_len, integer_observations)

    env.close()


def run_lander(exp_name="dqn-lander"):
    run(exp_name, "LunarLander-v2", 1, False)


def run_pong(exp_name="dqn-pong"):
    run(exp_name, "PongNoFrameskip-v4", 4, True)


if __name__ == "__main__":
    options = {}
    options['lander'] = run_lander
    options['pong'] = run_pong

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=options.keys())

    args = parser.parse_args()

    options[args.experiment]()