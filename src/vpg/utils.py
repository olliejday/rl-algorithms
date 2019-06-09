import numpy as np
import tensorflow as tf


def normalise(x):
    """
    Make to zero mean and unit std deviation
    """
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


class GradientBatchTrainer:
    """
    Setup update op such that we can apply gradient in batches.
    Because we want to use large batch sizes, but this can be too big for GPU memory, so we
    want to compute gradients in batches then sum them to update.
    From: https://stackoverflow.com/questions/42156957/how-to-update-model-parameters-with-accumulated-gradients
    """
    def __init__(self, loss_op, learning_rate):
        """
        Sets up the gradient batching and averaging TF operations.
        :param loss_op: Loss to optimise
        :param learning_rate: learning rate to use for optimiser
        """
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
        # Compute gradients; grad_pairs contains (gradient, variable) pairs
        grad_pairs = optimizer.compute_gradients(loss_op, trainable_vars)
        # Create operations which add a variable's gradient to its accumulator.
        self.accumulate_ops = [
            accumulator.assign_add(
                grad
            ) for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)
            if grad is not None and var is not None
        ]
        # Update trainable variables by applying the accumulated gradients
        # divided by the counter. Note: apply_gradients takes in a list of
        # (grad, var) pairs
        self.train_step = optimizer.apply_gradients(
            [(accumulator, var) \
             for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]
        )
        # Accumulators must be zeroed once the accumulated gradient is applied.
        self.zero_ops = [
            accumulator.assign(
                tf.zeros_like(tv)
            ) for (accumulator, tv) in zip(accumulators, trainable_vars)
        ]

    def get_batches(self, data, batch_size):
        """
        Returns data split into an array of batches of size batch_size.
        """
        n = int(len(data) / batch_size)
        if len(data) % batch_size == 0:
            return [data[i:i + batch_size] for i in range(n)]
        else:
            return [data[i * batch_size:(i + 1) * batch_size] for i in range(n)] + [data[n * batch_size:]]

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
        sess.run(self.zero_ops)
        for fd in self.get_feed_dicts(feed_dict, batch_size):
            sess.run(self.accumulate_ops, feed_dict=fd)
        sess.run(self.train_step)


class VPGBuffer:
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