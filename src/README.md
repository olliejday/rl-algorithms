# Reinforcement Learning Algorithms Source

## Implementation Notes

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
* Proximal Policy Gradients (PPO)
    * Uses Generalised Advantage Estimation (GAE)


## Custom Models

To define custom networks, follow the example models in `models.py` for a given algorithm.
In practice most common models can follow the structure of the
provided models, just changing the specific layers used in them to define new networks.

For example, some algorithms use subclasses of `tensorflow.keras.Model` to define 
networks. For this you must define the model in `__init__()` or `build()` functions 
that will be called. You must then define procedures to return outputs of a model for
passed inputs to `call(inputs)`. More details of this subclassing can be found in 
Tensorflow documentation. 
