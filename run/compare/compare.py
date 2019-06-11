import os

from src.common.utils import plot_training_curves

here_dir = os.path.dirname(os.path.realpath(__file__))
run_dir = os.path.dirname(here_dir)


def compare_cartpole():
    # AC logs
    ac = os.path.join(run_dir, "ac", "experiments", "ac-cartpole")
    # VPG logs
    vpg = os.path.join(run_dir, "vpg", "experiments", "vpg-cartpole")
    plot_training_curves({"ac": ac, "vpg": vpg}, save_to=os.path.join(here_dir, "cartpole.png"))


def compare_inverted_pendulum():
    # AC logs
    ac = os.path.join(run_dir, "ac", "experiments", "ac-inverted-pendulum")
    # SAC
    sac = os.path.join(run_dir, "sac", "experiments", "sac-inverted-pendulum")
    # VPG logs
    vpg = os.path.join(run_dir, "vpg", "experiments", "vpg-inverted-pendulum")
    plot_training_curves({"ac": ac, "sac": sac, "vpg": vpg}, save_to=os.path.join(here_dir, "inverted-pendulum.png"))


def compare_lander():
    # compare LunarLanderContinuous-v2 (so don't compare DQN)
    # AC logs
    vpg = os.path.join(run_dir, "vpg", "experiments", "vpg-lander")
    # SAC logs
    sac = os.path.join(run_dir, "sac", "experiments", "sac-lander")
    plot_training_curves({"vpg": vpg, "sac": sac}, save_to=os.path.join(here_dir, "lander.png"))




if __name__ == "__main__":
#     # compare_cartpole()
#     compare_inverted_pendulum()
    compare_lander()