import gym
import os

from src.vpg import VanillaPolicyGradients
from utils import TrainingLogger, set_global_seeds
from models import cnn_small, fc_small, cnn_med


def train(env_name, exp_name, model_fn, debug=False, seed=1, n_iter=100, **kwargs):
    env = gym.make(env_name)

    # Set random seeds
    set_global_seeds(seed, debug)

    vpg = VanillaPolicyGradients(model_fn,
                                 env,
                                 **kwargs)

    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_path = os.path.join(root_dir, "experiments", exp_name)
    training_logger = TrainingLogger(experiments_path, ["Env_name: {}".format(env_name),
                                                                  "Model_fn: {}".format(model_fn.__name__),
                                                                  "Seed: {}".format(seed),
                                                                  str(kwargs),
                                                                  str(vpg)])

    for itr in range(n_iter):
        buffer = vpg.sample_trajectories(itr)

        # get one long sequence of observations and actions
        # we keep rewards in trajectories until we discount
        obs = buffer.get_obs()
        acs = buffer.get_acs()
        logprobs = buffer.get_logprobs()
        rwds = buffer.rwds

        q_n, adv_n = vpg.estimate_return(obs, rwds)
        entropy, kl = vpg.update_parameters(obs, acs, q_n, adv_n, logprobs)

        training_logger.log(itr, [sum(r) for r in rwds], [len(r) for r in rwds], entropy, kl)

    env.close()


def train_cartpole(exp_name="vpg-debug"):
    env_name = "CartPole-v0"
    model_fn = fc_small

    train(env_name, exp_name, model_fn, debug=True, nn_baseline=True, nn_baseline_fn=fc_small)


def train_inverted_pendulum(exp_name="vpg-debug"):
    env_name = "InvertedPendulum-v2"
    model_fn = fc_small

    train(env_name, exp_name, model_fn, debug=True, nn_baseline=True, nn_baseline_fn=fc_small, discrete=False)


def train_grid(exp_name="vpg-grid"):
    env_name = "snake-grid-v0"
    model_fn = cnn_med

    train(env_name, exp_name, model_fn, nn_baseline=True, nn_baseline_fn=cnn_small)


def train_coord(exp_name="vpg-coord"):
    env_name = "snake-coord-v0"
    model_fn = fc_small

    train(env_name, exp_name, model_fn, nn_baseline=False, nn_baseline_fn=fc_small)


def train_stacked(exp_name="vpg-stacked"):
    env_name = "snake-stacked-v0"
    model_fn = cnn_med

    train(env_name, exp_name, model_fn, nn_baseline=True, nn_baseline_fn=cnn_small)


if __name__ == "__main__":
    """
    Example usage:
    
    * Debugging
    
    env_name = "CartPole-v0"
    exp_name = "vpg-debug"
    model_fn = fc_small

    train(env_name, exp_name, model_fn)
    
    * Run snake
    
    env_name = "snake-grid-v0"
    exp_name = "vpg-exp"
    model_fn = cnn_small

    train(env_name, exp_name, model_fn)
    
    """
    # train_coord()
    # train_grid()
    train_stacked()
    # train_cartpole()
    # train_inverted_pendulum()
    # plot_experiment("vpg-grid")
