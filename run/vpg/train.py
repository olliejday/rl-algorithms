import gym
import os

from src.vpg.vpg import VanillaPolicyGradients
from src.vpg.utils import VPGTrainingLogger
from src.vpg.models import fc_small
from src.common.utils import set_global_seeds


def train(env_name, exp_name, model_fn, debug=False, seed=1, n_iter=100, **kwargs):
    """
    General training setup.
    :param env_name: Environment name to train on
    :param exp_name: Experiment name to save logs to
    :param model_fn: Model function to call to generate the policy model (see src.vpg.models)
    :param debug: debug flag for seeding reproducibility vs performance
    :param seed: seed to setup system
    :param n_iter: number of iterations to train for
    :param kwargs: arguments to pass to VPG model __init__
    """
    env = gym.make(env_name)

    # Set random seeds
    set_global_seeds(seed, debug)

    vpg = VanillaPolicyGradients(model_fn,
                                 env,
                                 **kwargs)

    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_path = os.path.join(root_dir, "experiments", exp_name)
    training_logger = VPGTrainingLogger(experiments_path, ["Env_name: {}".format(env_name),
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


if __name__ == "__main__":
    """
    Example usage:
    
    env_name = "CartPole-v0"
    exp_name = "vpg-debug"
    model_fn = fc_small

    train(env_name, exp_name, model_fn)
    """
    # train_cartpole()
    # train_inverted_pendulum()
