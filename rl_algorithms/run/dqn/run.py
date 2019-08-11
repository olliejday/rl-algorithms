import gym
import os
import argparse

from rl_algorithms.src.dqn.dqn import run_model
from rl_algorithms.src.dqn.utils import get_env
from rl_algorithms.src.dqn.atari_wrappers import wrap_deepmind
from rl_algorithms.src.dqn.models import DQNCNNModelKerasSmall, DQNFCModelKeras


def run(exp_name,
        env,
        model_class,
        seed=123,
        model_number=None,
        **kwargs):
    """
    General running setup for saved models.
    :param exp_name: experiment directory to look for models in
    :param env: environment to run model in
    :param model_class: class for model for DQN
    :param seed: which experiment seed to run model of
    :param model_number: which model to run
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    models_dir = os.path.join(root_dir, "experiments", exp_name, str(seed), "models")

    assert os.path.exists(models_dir), "Invalid experiment name, models directory does not exist for" \
                                       ": {}, at: {}".format(exp_name, models_dir)

    if model_number is None:
        # then get latest model
        model_files = os.listdir(models_dir)
        model_number = max([int(f.split(".")[0].split("-")[1]) for f in model_files])
    model_path = os.path.join(models_dir, "model-{}.h5".format(model_number))

    assert os.path.exists(model_path), "Invalid model number, models file does not exist for" \
                                       ": {}, at: {}".format(model_number, model_path)

    run_model(env, model_class, model_path, **kwargs)
    env.close()


def run_lander(exp_name="dqn-lander", seed=123, debug=True):
    if debug:
        print('Random seed = %d' % seed)
    env = gym.make("LunarLander-v2")
    env = get_env(env, seed, debug)
    run(exp_name, env, DQNFCModelKeras, seed=seed, frame_history_len=1, integer_observations=False)


def run_pong(exp_name="dqn-pong", seed=1, debug=False):
    if debug:
        print('Random seed = %d' % seed)
    env = gym.make("PongNoFrameskip-v4")
    env = get_env(env, seed, debug)
    env = wrap_deepmind(env)
    run(exp_name, env, DQNCNNModelKerasSmall, seed=seed, frame_history_len=4, integer_observations=True)


if __name__ == "__main__":
    options = {}
    options['lander'] = run_lander
    options['pong'] = run_pong

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=options.keys())

    args = parser.parse_args()

    options[args.experiment]()