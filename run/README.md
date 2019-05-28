# Reinforcement Learning Algorithms Experiments

## Introduction

Here is example code to run the algorithms on OpenAI Gym environments. Also
includes experiments on environments to show performance.

## Usage

Train model by setting up as seen in examples in train.py
eg. opt spec etc. as quite complex options to tweak learning rate and exploration
schedules.
Then run with

python3 -m run.dqn.train

Plot training logs with

python3 -m run.dqn.plot dqn_lander

Run trained model with

python3 -m run.dqn.run dqn_snake_grid snake-grid-v0