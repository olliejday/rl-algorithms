import os

from src.common.utils import plot_training_curves

here_dir = os.path.dirname(os.path.realpath(__file__))
run_dir = os.path.dirname(here_dir)


def compare_cartpole():
    # AC logs
    ac = os.path.join(run_dir, "ac", "experiments", "ac-cartpole")
    # VPG logs
    vpg = os.path.join(run_dir, "vpg", "experiments", "vpg-cartpole")
    # PPO logs
    ppo = os.path.join(run_dir, "ppo", "experiments", "ppo-cartpole")
    plot_training_curves({"ac": ac, "vpg": vpg, "ppo": ppo}, save_to=os.path.join(here_dir, "cartpole.png"))


def compare_inverted_pendulum():
    # AC logs
    ac = os.path.join(run_dir, "ac", "experiments", "ac-inverted-pendulum")
    # SAC
    sac = os.path.join(run_dir, "sac", "experiments", "sac-inverted-pendulum")
    # VPG logs
    vpg = os.path.join(run_dir, "vpg", "experiments", "vpg-inverted-pendulum")
    # PPO logs
    ppo = os.path.join(run_dir, "ppo", "experiments", "ppo-inverted-pendulum")
    plot_training_curves({"ac": ac, "sac": sac, "vpg": vpg, "ppo": ppo}, save_to=os.path.join(here_dir, "inverted-pendulum.png"))


def compare_lander():
    # compare LunarLanderContinuous-v2 (so don't compare DQN)
    # AC logs
    vpg = os.path.join(run_dir, "vpg", "experiments", "vpg-lander")
    # SAC logs
    sac = os.path.join(run_dir, "sac", "experiments", "sac-lander")
    # PPO
    ppo = os.path.join(run_dir, "ppo", "experiments", "ppo-lander")
    plot_training_curves({"vpg": vpg, "sac": sac, "ppo": ppo}, save_to=os.path.join(here_dir, "lander.png"))


def compare_half_cheetah():
    # AC logs
    ac = os.path.join(run_dir, "ac", "experiments", "ac-half-cheetah")
    # SAC
    sac = os.path.join(run_dir, "sac", "experiments", "sac-half-cheetah")
    # VPG logs
    vpg = os.path.join(run_dir, "vpg", "experiments", "vpg-half-cheetah")
    # PPO
    ppo = os.path.join(run_dir, "ppo", "experiments", "ppo-half-cheetah")
    plot_training_curves({"ac": ac, "sac": sac, "vpg": vpg, "ppo": ppo}, save_to=os.path.join(here_dir, "half-cheetah.png"))


if __name__ == "__main__":
    # compare_cartpole()
    compare_inverted_pendulum()
    # compare_lander()
    # compare_half_cheetah()