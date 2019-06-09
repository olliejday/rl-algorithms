import os
import argparse
from src.common.utils import plot_training_curves


def plot_experiment(exp_name, save):
    """
    Plots an experiment saved in logs.
    :param exp_name: experiment name to plot
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_dir = os.path.join(root_dir, "experiments", exp_name)

    plot_training_curves(experiments_dir, save=save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training logs.")
    parser.add_argument('experiment_name', help="Name of experiments to plot logs for. Must exist as a directory with "
                                                "experiment_name/logs/logs.txt")
    parser.add_argument('--save', '-s', help="Save the figure to the logs directory.", action="store_true")

    args = parser.parse_args()

    plot_experiment(args.experiment_name, args.save)
