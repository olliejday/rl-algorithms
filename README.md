# Reinforcement Learning Algorithms

## Introduction

A set of implementations of key reinforcement learning algorithms in Tensorflow. Comes with 
example experiments and code to run on OpenAI Gym environments.

See src/README.md for implementation details. 

See run/README.md for details of how to run and some example experiments.

## Algorithms

* Vanilla Policy Gradients (VPG) / REINFORCE
* Deep Q-Networks (DQN)
* Advantage Actor Critic (A2C)
* Soft Actor Critic (SAC)
* Proximal Policy Optimization (PPO)

## Dependencies

Install dependencies for this project.

```
cd rl-algorithms
pip3 install -r requirements.txt
```

## TODO

Move to TensorFlow 2.0.

## Sources:

This repo is built on research work:

[1] Policy Gradient Methods for Reinforcement Learning with Function Approximation, Sutton et al, 2000. Algorithm: VPG.<br/>
[2] Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013. Algorithm: DQN.<br/>
[3] Asynchronous Methods for Deep Reinforcement Learning, Mnih et al, 2016. Algorithm: A2C. <br/>
[4] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, 
Haarnoja et al, 2018. <br/>
[5] Proximal Policy Optimization Algorithms, Schulman et al, 2017. <br/>
[6] High-Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al, 2015. <br/>

And code from:

[A] [UC Berkeley CS294-112 Assignments](https://github.com/berkeleydeeprlcourse/homework)
