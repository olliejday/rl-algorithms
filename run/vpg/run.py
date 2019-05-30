import gym
import os
import argparse

from src.vpg.vpg import run_model
from src.common.utils import set_global_seeds


def run(exp_name,
        env_name,
        model_number=None,
        seed=123,
        debug=False):
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

    run_model(env, model_path)

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

    args = parser.parse_args()

    run(args.experiment_name,
        args.environment_name,
        model_number=args.model_number,
        debug=args.debug,
        seed=args.seed,
        )
