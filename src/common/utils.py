import os
import tensorflow as tf
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_keras_session(debug):
    """
    Sets up the keras backed TF session
    :param debug: if True then we use config for better reproducibility but slightly reduced performance,
    otherwise we use better performance (but GPU usage may mean imperfect reproducibility)
    """
    import keras.backend as keras_backend

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


def set_global_seeds(seed, debug):
    """
    Set seeds for reproducibility.
    :param seed: Seed
    :param debug: if True then we use config for better reproducibility but slightly reduced performance,
    otherwise we use better performance (but GPU usage may mean imperfect reproducibility).
    Passed to set_keras_session()
    """
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    set_keras_session(debug)


def plot_training_curves(log_path, save_to=""):
    """

    :param log_path:
    :param save_to:
    :return:
    """
    assert os.path.exists(log_path), "Invalid path for log file, does not exist: {}".format(log_path)

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
