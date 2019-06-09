import os
import tensorflow as tf
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style


matplotlib.style.use("seaborn")


def set_global_seeds(seed, debug):
    """
    Set seeds for reproducibility.
    :param seed: Seed
    :param debug: if True then we use config for better reproducibility but slightly reduced performance,
    otherwise we use better performance (but GPU usage may mean imperfect reproducibility).
    """
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    if debug:
        # single threads and no GPU for better reproducibility
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1,
                                      device_count={'GPU': 0})
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        tf.keras.backend.set_session(sess)


def plot_training_curves(experiment_paths, save_to=""):
    """

    :param experiment_paths: a list of paths to logs.txt to plot
    :param save: if True saves to `experiments_dir`/Figure.png
    :return:
    """
    data = []
    timesteps = []
    for exp_path in experiment_paths:
        df = pd.read_csv(exp_path, sep=", ", engine="python", index_col=False)
        # average returns
        data.append(df["MeanReturn"].values)
        # average timesteps since each worker may have different numbers
        timesteps.append(df["Timesteps"].values)

    mean_return = np.mean(data, axis=0)
    std_return = np.std(data, axis=0)
    timesteps = np.mean(timesteps, axis=0)
    plt.plot(timesteps, mean_return, label="Mean Return", color="tomato")
    plt.fill_between(timesteps, mean_return - std_return, mean_return + std_return,
                     alpha=0.3, label="Std Return", color="tomato")
    plt.title("Training Curves")
    plt.ylabel("Return")
    plt.xlabel("Timesteps")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend()
    if save_to != "":
        plt.savefig(save_to)
    plt.show()


class PGBuffer:
    """
    Stores a dataset of trajectories for Policy Gradients style algorithms
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
        self.logprobs = []
        self.ptr = -1

    def add(self, ob, ac, rwd, lgprb):
        """
        Add s, a, r and logprob to buffer for this trajectory
        :param ob: state or observation
        :param ac: action taked
        :param rwd: reward
        :param lgprb: log prob of action taken
        """

        self.length += 1

        self.obs[self.ptr].append(ob)
        self.acs[self.ptr].append(ac)
        self.rwds[self.ptr].append(rwd)
        self.logprobs[self.ptr].append(lgprb)

    def next(self):
        """
        End of a trajectory, setup for next trajectory.
        Also call to initialise.
        """
        self.obs.append([])
        self.acs.append([])
        self.rwds.append([])
        self.logprobs.append([])
        self.ptr += 1

    """
    Getters to return observations, actions and log probs for all trajectories concatenated into 
    one long sequence.
    """

    def get_obs(self):
        return np.concatenate([x for x in self.obs])

    def get_acs(self):
        return np.concatenate([x for x in self.acs])

    def get_logprobs(self):
        return np.concatenate([x for x in self.logprobs])


class TrainingLogger:
    """
    Logs metrics during training
    """
    def __init__(self, experiments_dir, log_cols, config=[""]):
        """
        Creates a training logger
        :param experiments_dir: directory to save logs to (in `experiments_dir`/logs), must not already exist
        :param log_cols: list of strings of items to log.
        Note Time, Timesteps and MeanReturn are logged so need not be added (but can be under these names).
        :param config: list of strings to write in separate config log eg. hyperparameters
        """
        log_dir = os.path.join(experiments_dir, "logs")
        os.makedirs(log_dir)
        self.log_dir = log_dir
        self.log_path = os.path.join(self.log_dir, "logs.txt")
        # write parameters etc to a config file
        with open(os.path.join(self.log_dir, "config.txt"), "a") as fh:
            for line in config:
                fh.write(line + "\n")
        # must log these
        core_logs = ["Time", "Timesteps", "MeanReturn"]
        for l in core_logs:
            # add core logs to start of log_cols
            if l in log_cols:
                log_cols.remove(l)
            if l not in log_cols:
                log_cols.insert(0, l)

        # write log columns to file
        with open(self.log_path, "a") as fh:
            fh.write(", ".join(log_cols) + "\n")

        self.log_cols = log_cols
        # we want the print_format_string to be like
        # key1: {key1}\n key2: {key2} ...
        self.print_format_string = "\n" + "\n".join([k + ": {" + k + "}" for k in log_cols]) + "\n"
        # we want the log_format_string to be like
        # {key1}, {key2} ...
        self.log_format_string = ", ".join(["{" + k + "}" for k in log_cols]) + "\n"

    def log(self, **kwargs):
        """
        Logs metrics during training. Logs to file and prints to screen.
        :param kwargs: the keyword args with keys as `self.log_cols` to print and log to file
        """
        for c in self.log_cols:
            assert c in kwargs, "Missing log entry for: {}".format(c)
        print(self.print_format_string.format(**kwargs))
        with open(self.log_path, "a") as fh:
            fh.write(self.log_format_string.format(**kwargs))
