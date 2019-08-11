import os
import subprocess
import sys

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
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
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


class GradientBatchTrainer:
    """
       Setup update op such that we can apply gradient in batches.
       Because we want to use large batch sizes, but this can be too big for GPU memory, so we
       want to compute gradients in batches then sum them to update.
       From: https://stackoverflow.com/questions/42156957/how-to-update-model-parameters-with-accumulated-gradients

       Really easy to use!

       FROM

       lr = 1e-3  # learning rate
       loss_op = ...  # loss operation
       update_op = tf.train.AdamOptimizer(lr).minimize(loss_op)
       ...
       sess.run(update_op, feed_dict={...})

       TO

       lr = 1e-3  # learning rate
       loss_op = ...  # loss operation
       updater = GradientBatchTrainer(loss_op, lr, model.trainable_variables)
       updater.train(feed_dict={...}, batch_size=grad_batch_size, sess=sess)


       ___

       Can also explicitly compute and apply the gradients in order to manipulate before updates, eg. clipping, or
       averaging across models.

       grads_and_vars = updater.compute_gradients(feed_dict={...}, batch_size=grad_batch_size, sess=sess)
       ...
       updater.apply_gradients(grads_and_vars, sess)


       ___

       Implementation note:

       tf.train.AdamOptimizer().apply_gradients(x)
       expects x to be a tf.Tensor. So we use a placeholder for gradients of the sizes
       of the trainable variables this GradientBatcher is to compute and update gradients for.
       The gradients can then be manipulated eg. clipped and then applied, without requiring this
       to all be done with tf ops.

       """

    def __init__(self, loss_op, learning_rate, trainable_vars=None, average_gradient_batches=True):
        """
        Sets up the gradient batching and averaging TF operations.
        :param loss_op: Loss to optimise
        :param learning_rate: learning rate to use for optimiser
        :param trainable_vars: trainable variables to update, if None uses tf.trainable_variables()
        :param average_gradient_batches: whether to average the gradient over batches, otherwise it is summed
        """
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Fetch a list of our network's trainable parameters.
        if trainable_vars is None:
            trainable_vars = tf.trainable_variables()
        # Create variables to store accumulated gradients
        accumulators = [
            tf.Variable(
                tf.zeros_like(tv.initialized_value()),
                trainable=False
            ) for tv in trainable_vars
        ]
        # Compute gradients; grad_pairs contains (gradient, variable) pairs
        grad_pairs = optimizer.compute_gradients(loss_op, var_list=trainable_vars)
        # gradient placeholders
        self.grads_ph = [(tf.placeholder(dtype=tf.float32, shape=v.get_shape()), v) for v in trainable_vars]
        # operation to apply self.grads_ph to update
        self.apply_grads_op = optimizer.apply_gradients(self.grads_ph)
        # Create operations which add a variable's gradient to its accumulator.
        self.accumulate_ops = [
            accumulator.assign_add(
                grad
            ) for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)
            if grad is not None and var is not None
        ]

        if average_gradient_batches:
            # Create a variable for counting the number of accumulations
            accumulation_counter = tf.Variable(0.0, trainable=False)
            # The final accumulation operation is to increment the counter
            self.accumulate_ops.append(accumulation_counter.assign_add(1.0))
            self.accumulation_gradients = [(accumulator / accumulation_counter, var) \
                                           for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]
        else:
            self.accumulation_gradients = [(accumulator, var) \
                                           for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]

        # Accumulators must be zeroed once the accumulated gradient is applied.
        self.zero_ops = [
            accumulator.assign(
                tf.zeros_like(tv)
            ) for (accumulator, tv) in zip(accumulators, trainable_vars)
        ]

    def get_batches(self, data, batch_size, min_batch=0.3):
        """
        Returns data split into an array of batches of size batch_size.
        With additional data in a smaller batch if more than batch_size * min_batch amount in there
        """
        # if too big a batch size then use all the data
        if batch_size > len(data):
            batch_size = len(data)
        n = int(len(data) / batch_size)
        batches = [data[i * batch_size:(i + 1) * batch_size] for i in range(n)]
        # add any extras if not exact batch size
        if len(data) / batch_size - int(len(data) / batch_size) > min_batch:
            batches += [data[n:]]
        return batches

    def get_feed_dicts(self, feed_dict, batch_size):
        """
        Takes a feed_dict, TF style dictionary with keys of TF placeholders and values as data.
        Yields a batch at a time in the same dictionary format (same keys) but where values are now a single batch.
        """
        # batch each placeholder's data
        for k, v in feed_dict.items():
            feed_dict[k] = self.get_batches(v, batch_size)

        # get a TF feed_dict style dictionary of a data batch for each placeholder
        def sort_name(elem):
            return elem.__str__()

        sorted_dict = list(zip(*sorted(feed_dict.items(), key=sort_name)))
        sorted_keys = sorted_dict[0]
        sorted_vals = sorted_dict[1]
        for x in zip(*sorted_vals):
            feed_dict = {}
            for i, k in enumerate(sorted_keys):
                feed_dict[k] = x[i]
            yield feed_dict

    def train(self, feed_dict, batch_size, sess):
        """
        The training call on the gradient batch trainer.
        :param feed_dict: TF style feed_dict with keys of TF placeholders and values as data.
        :param batch_size: batch size to batch data into (batch must fit into CPU/GPU memory)
        :param sess: TF session to run on.
        """
        grads_and_vars = self.compute_gradients(feed_dict, batch_size, sess)
        self.apply_gradients(grads_and_vars, sess)

    def compute_gradients(self, feed_dict, batch_size, sess):
        """
        Computes the gradients averaged over gradient sub-batches.
        :param feed_dict: TF style feed_dict with keys of TF placeholders and values as data.
        :param batch_size: gradient sub-batch size to input to model
        :param sess: TF session to run on.
        :return: gradients as list of numpy arrays
        """
        sess.run(self.zero_ops)
        for fd in self.get_feed_dicts(feed_dict, batch_size):
            sess.run(self.accumulate_ops, feed_dict=fd)
        gradients = sess.run([g for (g, v) in self.accumulation_gradients])  # variable storing the accum grads
        return gradients

    def apply_gradients(self, gradients, sess):
        """
        Applies gradients to update model
        :param gradients: gradients to apply for update (eg. returned from self.compute_gradients(...))
        :param sess: session to run in
        """
        sess.run(self.apply_grads_op,
                 feed_dict={placeholder[0]: grad for placeholder, grad in zip(self.grads_ph, gradients)})


def sync_and_average_gradients(comm, grads):
    """
    Sync and average the gradients between models on all processes using MPI.
    """
    sync_grads = comm.allreduce(np.array(grads))
    avg_grads = sync_grads / float(comm.Get_size())
    return avg_grads


def sync_params(tf_params, comm, rank, controller, sess):
    """
    Sync all tf parameters across MPI processes. Call this initially to sync them all, then the gradient averaging
    should take care of keeping them the same.
    """
    cur_params = np.array(sess.run(tf_params))
    new_params = comm.bcast(cur_params, root=controller)
    if rank != controller:
        assign_ops = [tf.assign(param, new_param) for param, new_param in zip(tf_params, new_params)]
        sess.run(assign_ops)


def mpi_fork(n, bind_to_core=False):
    """
    Re-launches the current script with workers linked by MPI.
    Also, terminates the original process that launched it.
    Taken almost without modification from the Baselines function of the
    `same name`_.
    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py
    Args:
        n (int): Number of process to split into.
        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n <= 1:
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()
