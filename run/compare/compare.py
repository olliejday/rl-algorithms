import os

from src.common.utils import plot_training_curves


def compare_cartpole():
    here_dir = os.path.dirname(os.path.realpath(__file__))
    run_dir = os.path.dirname(here_dir)
    # AC logs
    ac = os.path.join(run_dir, "ac", "experiments", "ac-cartpole")
    # VPG logs
    vpg = os.path.join(run_dir, "vpg", "experiments", "vpg-cartpole")

    plot_training_curves({"ac": ac, "vpg": vpg})


if __name__ == "__main__":
    compare_cartpole()