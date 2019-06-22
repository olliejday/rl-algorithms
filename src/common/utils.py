import os
import tensorflow as tf
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style
from matplotlib import cm


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


def plot_training_curves(experiments, save_to="", title="Training Curves"):
    """

    :param experiments: a dict of
    keys = labels,
    values = paths to experiment directories to plot.
    An experiment directory is expected to have experiment_dir/1/logs/logs.txt, experiment_dir/2/logs/logs.txt, ...
    where 1, 2, ... are the seeds run for that experiment that will be averaged over
    :param save: if True saves to `experiments_dir`/Figure.png
    :return:
    """
    # list of colours to plot in
    cols = cm.tab10.colors

    for i, (label, experiment) in enumerate(experiments.items()):
        # average over the experiment seeds
        data = []
        timesteps = []
        seeds = os.listdir(experiment)
        for seed in seeds:
            log_dir = os.path.join(experiment, str(seed), "logs")
            if not os.path.isdir(log_dir):
                # only want directories
                continue
            log_path = os.path.join(log_dir, "logs.txt")
            df = pd.read_csv(log_path, sep=", ", engine="python", index_col=False)
            # average returns
            data.append(df["MeanReturn"].values)
            # average timesteps since each worker may have different numbers
            timesteps.append(df["Timesteps"].values)
        # plot this experiment's averaged data
        mean_return = np.mean(data, axis=0)
        std_return = np.std(data, axis=0)
        timesteps = np.mean(timesteps, axis=0)
        plt.plot(timesteps, mean_return, color=cols[i], label=label)
        plt.fill_between(timesteps, mean_return - std_return, mean_return + std_return,
                         alpha=0.3, color=cols[i])

    plt.title(title)
    plt.ylabel("Mean Episode Return")
    plt.xlabel("Timesteps")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend(loc="best")

    if save_to != "":
        plt.savefig(save_to)
    plt.show()


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
