import numpy as np


def normalise(x):
    """
    Make to zero mean and unit std deviation
    """
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


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