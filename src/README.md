# Reinforcement Learning Algorithms Source

Some implementation notes on the included algorithms:

* Vanilla Policy Gradients (VPG) / REINFORCE
    * Advantage normalisation to reduce variance and reward to go usually help training
    * Larger batch sizes can work better, so we use a GradeintBatcher to allow 
    gradients to be summed over larger batches than fits into GPU memory at once.
    * Sensitive to learning rate
    * Using a Neural Network baseline can reduce variance and improve training
* Deep Q-Networks (DQN)
    * Also included Double DQN variant
* Advantage Actor Critic (AC)
    * Vanilla A2C version
* Soft Actor Critic (SAC)
    * Continuous actions only
    * Reparamaterised or REINFORCE style gradients
    * Two Q functions for greater stability (similar to Double DQN)
