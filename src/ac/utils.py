import numpy as np
import tensorflow as tf
import matplotlib as mpl
import time
import os

from src.common.utils import plot_training_curves


mpl.style.use("seaborn")


def normalise(x):
    """
    Make to zero mean and unit std deviation
    """
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


class ACBuffer:
    """
    Stores a dataset of trajectories
    """

    def __init__(self):
        """
        Starts uninitialised to -1s and empty lists.
        Call next() to initialise before use.
        """
        self.length = -1
        self.obs = []
        self.acs = []
        self.rwds = []
        self.next_obs = []
        self.terminals = []
        self.logprobs = []
        self.ptr = -1

    def add(self, ob, ac, rwd, next_ob, done, lgprb):
        """
        Add s, a, r, s', done and logprob to buffer for this trajectory
        :param ob: state or observation
        :param ac: action taked
        :param rwd: reward
        :param next_ob: next state or observation
        :param lgprb: log prob of action taken
        """

        self.length += 1

        self.obs[self.ptr].append(ob)
        self.acs[self.ptr].append(ac)
        self.rwds[self.ptr].append(rwd)
        self.next_obs[self.ptr].append(next_ob)
        self.terminals[self.ptr].append(done)
        self.logprobs[self.ptr].append(lgprb)

    def next(self):
        """
        End of a trajectory, setup for next trajectory.
        Also call to initialise.
        """
        self.obs.append([])
        self.acs.append([])
        self.rwds.append([])
        self.next_obs.append([])
        self.terminals.append([])
        self.logprobs.append([])
        self.ptr += 1

    def get(self):
        """
        Getter to return observations, actions, rewards, next observations, terminals and log probs for all
        trajectories concatenated into one long sequence.
        """
        return np.concatenate([x for x in self.obs]), \
               np.concatenate([x for x in self.acs]), \
               np.concatenate([x for x in self.rwds]),\
               np.concatenate([x for x in self.next_obs]), \
               np.concatenate([x for x in self.terminals]),\
               np.concatenate([x for x in self.logprobs])


class ACTrainingLogger:
    """
    Logs metrics during training
    """
    def __init__(self, experiments_dir, initial_logs=[""], do_plot=False):
        """
        Creates a training logger
        :param log_dir: directory to save logs to, must not already exist
        :param initial_logs: list of strings to write in separate config log eg. hyperparameters
        """
        # for tracking progress
        self.timesteps = 0
        # for plotting
        self.do_plot = do_plot

        log_dir = os.path.join(experiments_dir, "logs")
        assert not os.path.exists(
            log_dir), "Log dir %s already exists! Delete it first or use a different dir" % log_dir
        os.makedirs(log_dir)
        self.log_dir = log_dir
        self.log_path = os.path.join(self.log_dir, "logs.txt")
        self.plot_path = os.path.join(self.log_dir, "figure.png")
        # write parameters etc to a config file
        with open(os.path.join(self.log_dir, "config.txt"), "a") as fh:
            for line in initial_logs:
                fh.write(line + "\n")
        with open(self.log_path, "a") as fh:
            fh.write("Time, Iteration, MeanReturn, StdReturn, MaxReturn, MinReturn, PolicyEntopy, KL, "
                     "EpLenMean, EpLenStd, Timesteps\n")

    def log(self, itr, returns, ep_lengths, entropy, kl):
        """
        Logs metrics during training. Logs to file and prints to screen.
        :param itr: Current iteration of training
        :param returns: A set of returns from episodes during training since last log.
        :param ep_lengths: A set of episode lengths of episodes during training since last log.
        :param entropy: the current entropy of the policy
        :param kl: KL divergence of the updated policy from the old policy.
        """
        self.timesteps += np.sum(ep_lengths)

        print("{}, Iteration: {}\n MeanReturn: {:.3f}\n StdReturn: {:.3f}\n MaxReturn: {:.3f}\n MinReturn: {:.3f}\n "
              "PolicyEntopy: {:.3f}\n KL: {:.3f}\n EpLenMean: {:.3f}\n EpLenStd: {:.3f}\n Timesteps: {}\n".format(
                time.strftime("%d/%m/%Y %H:%M:%S"),
                itr,
                np.mean(returns),
                np.std(returns),
                np.max(returns),
                np.min(returns),
                np.mean(entropy),
                np.mean(kl),
                np.mean(ep_lengths),
                np.std(ep_lengths),
                self.timesteps))
        with open(self.log_path, "a") as fh:
            fh.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(time.strftime("%d/%m/%Y %H:%M:%S"),
                                                                           itr,
                                                                           np.mean(returns),
                                                                           np.std(returns),
                                                                           np.max(returns),
                                                                           np.min(returns),
                                                                           np.mean(entropy),
                                                                           np.mean(kl),
                                                                           np.mean(ep_lengths),
                                                                           np.std(ep_lengths),
                                                                           self.timesteps))

        if self.do_plot:
            plot_training_curves(self.log_path, save_to=self.plot_path)


def gaussian_log_likelihood(x, mu, log_std):
    """
    Computes the log probability of x under a Gaussian with mean mu and log_std
    """
    std = tf.exp(log_std) + 1e-8
    return - 0.5 * tf.reduce_sum(((mu - x) / std) ** 2 + 2 * log_std + np.log(2 * np.pi), axis=1)


def gather_nd(x, inds, name="gather_nd"):
    """
    For ith row of x, gathers the inds[i] element.
    """
    indices = tf.stack([tf.range(tf.shape(inds)[0]), inds], axis=1)
    return tf.gather_nd(x, indices, name=name)
