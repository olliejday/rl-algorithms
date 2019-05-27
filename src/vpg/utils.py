import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import time
import os
import random
import keras.backend as keras_backend

mpl.style.use("seaborn")


def normalise(x):
    """
    Make to zero mean and unit std deviation
    """
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


class VPGBuffer:
    """
    Stores a dataset of trajectories
    """

    def __init__(self):
        self.length = 0
        self.obs = [[]]
        self.acs = [[]]
        self.rwds = [[]]
        self.logprobs = [[]]
        self.ptr = 0

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
        End of a trajectory, setup for next trajectory
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


# TODO: comment this logging
class TrainingLogger:
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


def plot_experiment(exp_name):
    root_dir = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(root_dir, "experiments/{}/logs/logs.txt".format(exp_name))
    plot_training_curves(log_path)


def plot_training_curves(log_path, save_to=""):
    df = pd.read_csv(log_path, sep=", ", engine="python")
    plt.plot(df["Timesteps"], df["MeanReturn"], label="Mean Return", color="tomato")
    plt.fill_between(df["Timesteps"], df["MeanReturn"] - df["StdReturn"], df["MeanReturn"] + df["StdReturn"],
                     alpha=0.3, label="Std Return", color="tomato")
    plt.title("Training Curves")
    plt.ylabel("Return")
    plt.xlabel("Timesteps")
    plt.legend()
    if save_to != "":
        plt.savefig(save_to)
    plt.show()


# TODO: comment this class
class GradientBatchTrainer:
    # setup update op such that we can apply gradient in batches
    # from: https://stackoverflow.com/questions/42156957/how-to-update-model-parameters-with-accumulated-gradients
    def __init__(self, loss_op, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Fetch a list of our network's trainable parameters.
        trainable_vars = tf.trainable_variables()
        # Create variables to store accumulated gradients
        accumulators = [
            tf.Variable(
                tf.zeros_like(tv.initialized_value()),
                trainable=False
            ) for tv in trainable_vars
        ]
        # Create a variable for counting the number of accumulations
        accumulation_counter = tf.Variable(0.0, trainable=False)
        # Compute gradients; grad_pairs contains (gradient, variable) pairs
        grad_pairs = optimizer.compute_gradients(loss_op, trainable_vars)
        # Create operations which add a variable's gradient to its accumulator.
        self.accumulate_ops = [
            accumulator.assign_add(
                grad
            ) for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)
            if grad is not None and var is not None
        ]
        # The final accumulation operation is to increment the counter
        self.accumulate_ops.append(accumulation_counter.assign_add(1.0))
        # Update trainable variables by applying the accumulated gradients
        # divided by the counter. Note: apply_gradients takes in a list of
        # (grad, var) pairs
        self.train_step = optimizer.apply_gradients(
            [(accumulator / accumulation_counter, var) \
             for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]
        )
        # Accumulators must be zeroed once the accumulated gradient is applied.
        self.zero_ops = [
            accumulator.assign(
                tf.zeros_like(tv)
            ) for (accumulator, tv) in zip(accumulators, trainable_vars)
        ]
        # Add one last op for zeroing the counter
        self.zero_ops.append(accumulation_counter.assign(0.0))

    def get_batches(self, data, batch_size):
        n = int(len(data) / batch_size)
        if len(data) % batch_size == 0:
            return [data[i:i + batch_size] for i in range(n)]
        else:
            return [data[i * batch_size:(i + 1) * batch_size] for i in range(n)] + [data[n * batch_size:]]

    def get_feed_dicts(self, feed_dict, batch_size):
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
        sess.run(self.zero_ops)
        for fd in self.get_feed_dicts(feed_dict, batch_size):
            sess.run(self.accumulate_ops, feed_dict=fd)
        sess.run(self.train_step)


def set_keras_session(debug):
    """
    Sets up the keras backed TF session
    :param debug: if true then we use config for better reproducibility but slightly reduced performance,
    otherwise we use better performance (but GPU usage may mean imperfect reproducibility)
    """
    if debug:
        # single threads and no GPU for better reproducibility
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1,
                                      device_count={'GPU': 0})
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        keras_backend.set_session(sess)
    else:
        # otherwise we allow GPU usage for quicker training but poorer reproducibility
        sess = tf.Session(graph=tf.get_default_graph())
        keras_backend.set_session(sess)


def set_global_seeds(i, debug):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(i)
    random.seed(i)
    tf.set_random_seed(i)
    set_keras_session(debug)