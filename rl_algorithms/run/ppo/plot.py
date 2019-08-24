import os
import argparse
from rl_algorithms.src.common.utils import plot_training_curves


def plot_experiment(exp_names, save):
    """
    Plots an experiment saved in logs.
    :param exp_name: list of experiment name to plot
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))

    save_to = ""
    if save:
        if len(exp_names) == 1:
            save_to = os.path.join(root_dir, "experiments", exp_names[0], "Figure.png")
        else:
            save_to = os.path.join(root_dir, "experiments", "Figure.png")

    exps = {exp_name: os.path.join(root_dir, "experiments", exp_name) for exp_name in exp_names}

    plot_training_curves(exps, save_to=save_to, title="Training Curves {}".format("PPO"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training logs.")
    parser.add_argument('experiment_name', help="List of names of experiments to plot logs for. Must exist as"
                                                " directories with experiment_name/logs/logs.txt", nargs="+")
    parser.add_argument('--save', '-s', help="Save the figure to the logs directory.", action="store_true")

    args = parser.parse_args()

    plot_experiment(args.experiment_name, args.save)
