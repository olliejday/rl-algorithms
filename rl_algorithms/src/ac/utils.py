import numpy as np


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
