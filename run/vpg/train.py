import gym
import os
import argparse

from src.vpg.vpg import VanillaPolicyGradients
from src.vpg.utils import VPGTrainingLogger
from src.vpg.models import fc_small, fc_medium, cnn_small, cnn_medium
from src.common.utils import set_global_seeds


def train(env, exp_name, model_fn, n_iter=100, save_every=25, **kwargs):
    """
    General training setup.
    :param env: Environment to train on
    :param exp_name: Experiment name to save logs to
    :param model_fn: Model function to call to generate the policy model (see src.vpg.models)
    :param n_iter: number of iterations to train for
    :param save_every: number of iterations to save models at
    :param kwargs: arguments to pass to VPG model __init__
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_path = os.path.join(root_dir, "experiments", exp_name)

    vpg = VanillaPolicyGradients(model_fn,
                                 env,
                                 experiments_path=experiments_path,
                                 **kwargs)

    training_logger = VPGTrainingLogger(experiments_path, ["Model_fn: {}".format(model_fn.__name__),
                                                           str(vpg)])

    vpg.setup_graph()

    for itr in range(1, n_iter+1):
        buffer = vpg.sample_trajectories(itr)

        # get one long sequence of observations and actions
        # we keep rewards in trajectories until we discount
        obs = buffer.get_obs()
        acs = buffer.get_acs()
        logprobs = buffer.get_logprobs()
        rwds = buffer.rwds

        q_n, adv_n = vpg.estimate_return(obs, rwds)
        approx_entropy, approx_kl = vpg.update_parameters(obs, acs, q_n, adv_n, logprobs)

        training_logger.log(itr, [sum(r) for r in rwds], [len(r) for r in rwds], approx_entropy, approx_kl)

        if itr % save_every == 0:
            vpg.save_model(training_logger.timesteps)

    env.close()


def train_cartpole(seed=123, debug=True, exp_name="vpg-cartpole"):
    env = gym.make("CartPole-v0")
    set_global_seeds(seed, debug)
    env.seed(seed)
    train(env, exp_name, fc_small, nn_baseline=True, nn_baseline_fn=fc_small, min_timesteps_per_batch=2500,
          learning_rate=0.01, n_iter=30, render_every=1000)


def train_inverted_pendulum(seed=123, debug=True, exp_name="vpg-inverted-pendulum"):
    import roboschool
    env = gym.make("RoboschoolInvertedPendulum-v1")
    set_global_seeds(seed, debug)
    env.seed(seed)
    train(env, exp_name, fc_small, nn_baseline=True, nn_baseline_fn=fc_small, min_timesteps_per_batch=2500,
          discrete=False, learning_rate=0.05, n_iter=30, gamma=0.9, render_every=1000)


def train_lander(seed=123, debug=False, exp_name="vpg-lander"):
    env = gym.make("LunarLanderContinuous-v2")
    set_global_seeds(seed, debug)
    env.seed(seed)
    train(env, exp_name, fc_small, nn_baseline=True, nn_baseline_fn=fc_small,
          discrete=False, min_timesteps_per_batch=40000, learning_rate=0.005, gradient_batch_size=40000)


def train_pong(seed=123, debug=False, exp_name="vpg-pong"):
    env = gym.make("Pong-v0")
    set_global_seeds(seed, debug)
    env.seed(seed)
    train(env, exp_name, cnn_small, nn_baseline=True, nn_baseline_fn=cnn_small,
          discrete=True, min_timesteps_per_batch=10000, learning_rate=0.005)


if __name__ == "__main__":
    options = {}
    options['lander'] = train_lander
    options['inverted-pendulum'] = train_inverted_pendulum
    options['cartpole'] = train_cartpole
    options['pong'] = train_pong

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=options.keys())

    args = parser.parse_args()

    options[args.experiment]()