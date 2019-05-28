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
        debug=False,
        **kwargs):
    """
    General running setup for saved models.
    :param exp_name: experiment directory to look for models in
    :param env_name: environment to run model in
    :param frame_history_len: how many observations to stack in a sequence to pass to model
    :param integer_observations: whether to store integer observations in replay buffer
    :param model_number: model number to load, if None then latest model is loaded
    :param seed: seed to set for system
    :param debug: debug flag for seeding reproducibility vs performance
    :param kwargs: any kwargs to pass to DQN __init__
    :return:
    """
    print('Random seed = %d' % seed)
    env = gym.make(env_name)
    env = get_env(env, seed, debug)

    root_dir = os.path.dirname(os.path.realpath(__file__))
    models_dir = os.path.join(root_dir, "experiments", exp_name, "models")
    if model_number is None:
        # then get latest model
        model_files = os.listdir(models_dir)
        model_number = max([int(f.split(".")[0].split("-")[1]) for f in model_files])
    model_path = os.path.join(models_dir, "model-{}.h5".format(model_number))

    run_model(env, model_path, frame_history_len, integer_observations)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training logs.")
    experiment_args = parser.add_argument_group("Experiment Arguments", "Parameters for running experiments.")
    experiment_args.add_argument('experiment_name', help="Name of experiments to run model from. Must exist as a directory with "
                                                "experiment_name/models/model-xxxx.txt")
    experiment_args.add_argument('environment_name', help="Name of OpenAI Gym environment to run in.")
    experiment_args.add_argument('--model_number', '-m', help="Model number xxxx Must exist as a directory with "
                                             "experiment_name/models/model-xxxx.txt. If None then latest is used")
    experiment_args.add_argument('--seed', '-s', help="Seed to set for system.", default=123)
    experiment_args.add_argument('--debug', '-d', help="Whether to use debugging for reproducibility but reduced performance.",
                        action="store_true")
    model_args = parser.add_argument_group("Model Arguments", "Parameters for model to run.")
    # settings for DQN model
    model_args.add_argument('--integer_observations', '-int', help="If True, expects integer observations.", default=True)
    model_args.add_argument('--frame_history_len', '-f', help="Number of frames to stack observations into sequence.",
                        default=4)

    args = parser.parse_args()

    run(args.experiment_name,
        args.environment_name,
        args.frame_history_len,
        args.integer_observations,
        model_number=args.model_number,
        debug=args.debug,
        seed=args.seed,
        )
