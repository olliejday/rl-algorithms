import gym
import os
import argparse
import roboschool

from src.ac.ac import ActorCrtic
from src.ac.utils import ACTrainingLogger
from src.ac.models import fc
from src.common.utils import set_global_seeds


def train(env, exp_name, model_fn, n_iter=100, save_every=25, **kwargs):
    """
    General training setup.
    :param env: Environment to train on
    :param exp_name: Experiment name to save logs to
    :param model_fn: Model function to call to generate the policy model (see src.ac.models)
    :param n_iter: number of iterations to train for
    :param save_every: number of iterations to save models at
    :param kwargs: arguments to pass to AC model __init__
    """
    root_dir = os.path.dirname(os.path.realpath(__file__))
    experiments_path = os.path.join(root_dir, "experiments", exp_name)

    ac = ActorCrtic(model_fn,
                     env,
                     experiments_path=experiments_path,
                     **kwargs)

    training_logger = ACTrainingLogger(experiments_path, ["Model_fn: {}".format(model_fn.__name__),
                                                          str(ac)])

    ac.setup_graph()

    for itr in range(1, n_iter+1):
        buffer = ac.sample_trajectories(itr)

        # get one long sequence of observations and actions
        # we keep rewards in trajectories until we discount
        obs, acs, rwds, next_obs, terminals, logprobs = buffer.get()

        ac.update_parameters(obs, next_obs, rwds, terminals, acs)
        approx_entropy, approx_kl = ac.training_metrics(obs, acs, logprobs)

        training_logger.log(itr, [sum(r) for r in buffer.rwds], [len(r) for r in buffer.rwds], approx_entropy, approx_kl)

        if itr % save_every == 0:
            ac.save_model(training_logger.timesteps)

    env.close()


def train_cartpole(seed=123, debug=True, exp_name="ac-cartpole"):
    env = gym.make("CartPole-v0")
    set_global_seeds(seed, debug)
    env.seed(seed)
    train(env, exp_name, fc, min_timesteps_per_batch=2000,
          n_iter=30, render_every=1000, num_target_updates=10,
          num_grad_steps_per_target_update=10)


def train_inverted_pendulum(seed=123, debug=True, exp_name="ac-inverted-pendulum"):
    env = gym.make("RoboschoolInvertedPendulum-v1")
    set_global_seeds(seed, debug)
    env.seed(seed)
    train(env, exp_name, fc, min_timesteps_per_batch=2500,
          discrete=False, learning_rate=0.05, n_iter=30, gamma=0.9, render_every=1000)


def train_half_cheetah(seed=123, debug=False, exp_name="ac-half-cheetah"):
    env = gym.make("RoboschoolHalfCheetah-v1")
    set_global_seeds(seed, debug)
    env.seed(seed)
    train(env, exp_name, fc,
          discrete=False, min_timesteps_per_batch=50000, learning_rate=0.005, gradient_batch_size=3000,
          render_every=1000)

if __name__ == "__main__":
    options = {}
    options['inverted-pendulum'] = train_inverted_pendulum
    options['cartpole'] = train_cartpole
    options['half-cheetah'] = train_half_cheetah

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=options.keys())

    args = parser.parse_args()

    options[args.experiment]()