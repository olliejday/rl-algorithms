import gym
import os
import argparse
import gym_snake

from src.dqn.dqn import run_model
from src.dqn.utils import get_env


def run(exp_name,
        env,
        frame_history_len,
        integer_observations,
        model_number=None):
    """
    General running setup for saved models.
    :param exp_name: experiment directory to look for models in
    :param env: environment to run model in
    :param frame_history_len: how many observations to stack in a sequence to pass to model
    :param integer_observations: whether to store integer observations in replay buffer
    :param model_number: model number to load, if None then latest model is loaded
    """
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


def run_snake(exp_name="dqn-snake-{}", env_type="grid", seed=123, debug=False):
    if debug:
        print('Random seed = %d' % seed)

    exp_name = exp_name.format(env_type)

    env = gym.make("snake-{}-v0".format(env_type))
    env = get_env(env, seed, debug)

    run(exp_name, env, 4, True)


if __name__ == "__main__":
    options = {}
    options['snake'] = run_snake

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=options.keys())

    args = parser.parse_args()

    options[args.experiment]()