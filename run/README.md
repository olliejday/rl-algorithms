# Reinforcement Learning Algorithms Experiments

## Introduction

Here is example code to run the algorithms on OpenAI Gym environments. Also
includes experiments on environments to show performance.

## Usage

## Training

Train model by setting up as seen in examples in train.py
eg. opt spec etc. as quite complex options to tweak learning rate and exploration
schedules.

## Running and plotting

After training a model, running a saved trained model can be done as:

```python3 -m run.<algorithm>.run <experiment_name> <environment_name>```

Where :

`<algorithm>` is the algorithm eg. `dqn` or `vpg`<br/>
`<experiment_name>` is the name of the experiment (ie. the top directory storing models) eg. `dqn_lander`<br/>
`<environment_name>` is the name of the environment to run in eg. `LunarLander-v2`<br/>


Run 

```python3 -m run.<algorithm>.run -h```

for more information about options and arguments.

<br/>

Similarly plotting the training curves from training logs can be done:

```python3 -m run.<algorithm>.plot <experiment_name>```

Where:
 
`<algorithm>` is the algorithm eg. `dqn` or `vpg`.<br/>
`<experiment_name>` is the name of the experiment (ie. the top directory storing models) eg. `dqn_lander`<br/>


Run 

```python3 -m run.<algorithm>.plot -h```

for more information about options and arguments.