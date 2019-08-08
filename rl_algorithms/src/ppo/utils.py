import numpy as np


def normalise(x):
    """
    Make to zero mean and unit std deviation
    """
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def reward_to_go(rwds, gamma):
    """
    Computes discounted sum of rewards.
    Uses reward to go, ie. each timestep's return is the sum of discounted rewards from
    then until end.

    :param rwds: a list returns
    :param gamma: discount parameter
    :return: discounted reward sequences
    """
    # discount vector
    discount = np.array([gamma ** i for i in range(len(rwds)+1)])
    rwds_to_go = [np.dot(discount[:-i-1], rwds[i:]) for i in range(len(rwds))]
    return rwds_to_go


class PPOBuffer:
    """
    Stores a dataset of trajectories for Policy Gradients style algorithms
    """

    def __init__(self):
        """
        Setup empty buffer
        """
        self.length = 0
        self.obs = [[]]
        self.acs = [[]]
        self.rwds = [[]]
        self.logprobs = [[]]
        self.vals = [[]]  # value function estimates
        self.ptr = 0

    def add(self, ob, ac, rwd, lgprb, val):
        """
        Add s, a, r and logprob to buffer for this trajectory
        :param ob: state or observation
        :param ac: action taked
        :param rwd: reward
        :param lgprb: log prob of action taken
        :param val: value function estimate for this state
        """

        self.length += 1

        self.obs[self.ptr].append(ob)
        self.acs[self.ptr].append(ac)
        self.rwds[self.ptr].append(rwd)
        self.logprobs[self.ptr].append(lgprb)
        self.vals[self.ptr].append(val)

    def next(self):
        """
        End of a trajectory, setup for next trajectory.
        """
        self.obs.append([])
        self.acs.append([])
        self.rwds.append([])
        self.logprobs.append([])
        self.vals.append([])
        self.ptr += 1

    def done(self):
        """
        If we finish an episode, add a 0 last value for terminal state.
        """
        self.rwds[self.ptr].append(0)
        self.vals[self.ptr].append(0)

    def early_stop(self, val):
        """
        If we finish an episode early, add a value function estimate to bootstrap the rest
        of the episode.
        """
        self.rwds[self.ptr].append(val)
        self.vals[self.ptr].append(val)

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

    def get_gae(self, gamma, gae_lambda):
        """
        Computes the discounted rewards to go, and the normalised GAE advantage estimates.
        :param gamma: discount rate
        :param gae_lambda: GAE lambda parameter
        :return: rewards, advantages
        """
        deltas = [self.rwds[i][:-1] + gamma * np.array(self.vals[i][1:]) - self.vals[i][:-1] for i in range(len(self.rwds))]
        advs = normalise(np.hstack([reward_to_go(delta, gamma * gae_lambda) for delta in deltas]))
        # we want to include the terminal value reward / estimate in the discount computation but not
        # include it as a reward itself for the updates.
        rwds = np.hstack([reward_to_go(rwd, gamma)[:-1] for rwd in self.rwds])

        return rwds, advs

    def extend(self, buffer):
        """
        Extends this buffer with another.
        """
        # if we have an empty buffer we fill it
        if self.length == 0:
            self.length = buffer.length
            self.obs = buffer.obs
            self.acs = buffer.acs
            self.rwds = buffer.rwds
            self.logprobs = buffer.logprobs
            self.vals = buffer.vals

        # otherwise we extend it
        else:
            self.length += buffer.length
            self.obs.extend(buffer.obs)
            self.acs.extend(buffer.acs)
            self.rwds.extend(buffer.rwds)
            self.logprobs.extend(buffer.logprobs)
            self.vals.extend(buffer.vals)
