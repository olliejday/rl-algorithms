import os
import argparse
from src.common.utils import plot_training_curves


def plot_experiment(exp_name):
    """
    Plots an experiment saved in logs.
    :param exp_name: experiment name to plot
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(root_dir, "experiments/{}/logs/logs.txt".format(exp_name))

    assert os.path.exists(log_path), "Invalid experiment name, logs do not exist for: {}, at: {}".format(exp_name,
                                                                                                         log_path)

    plot_training_curves(log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training logs.")
    parser.add_argument('experiment_name', help="Name of experiments to plot logs for. Must exist as a directory with "
                                                "experiment_name/logs/logs.txt")

    args = parser.parse_args()

    plot_experiment(args.experiment_name)
