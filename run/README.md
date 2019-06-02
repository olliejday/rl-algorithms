# Reinforcement Learning Algorithms Experiments

## Introduction

Here is example code to run the algorithms on OpenAI Gym environments. Also
includes experiments on environments to show performance.

## Examples

This snake branch includes experiments for the [snake](https://github.com/olliejday/snake-rl) environments. Included
is code to run [snake-grid-v0](https://github.com/olliejday/snake-rl) and [snake-stacked-v0](https://github.com/olliejday/snake-rl).
We also have trained models, training logs and plots for [snake-grid-v0](https://github.com/olliejday/snake-rl) experiments 
with the DQN algorithm.

## Usage

For each algorithm we have a template for training, running and plotting interface.

All examples are OpenAI Gym environments, but some included are Roboschool and Atari environments
which require installing too.

## Training

Train model by setting up a training function as seen in examples in `run/<algorithm>/train.py`. You need to 
provide hyperparamers, environment name etc.

Then add this custom training function, `train_fn` to the options dictionary in `train.py`. You must 
give it a name `<experiment>`.

```options['<experiment>'] = <train_fn>```  

Then to train this custom training setup, call

```python3 -m run.<algorithm>.train <experiment>```

eg. ```python3 -m run.dqn.train lander```

## Running

After training a model, running a saved trained model can be done as:

```python3 -m run.<algorithm>.run <experiment>```

Where `<experiment>` is the name of the experiment created above.

For custom trained models, you will have to add a running function, `run_fn`, similarly to custom training functions 
above. See the examples for details, most hyperparameters should be copied from the training function.

eg. ```python3 -m run.dqn.run lander```

## Plotting

Plotting the training curves from training logs can be done:

```python3 -m run.<algorithm>.plot <experiment_name>```

Where:
 
`<algorithm>` is the algorithm eg. `dqn` or `vpg`.<br/>
`<experiment_name>` is the name of the experiment (ie. the top directory storing models) eg. `dqn_lander`<br/>


Run 

```python3 -m run.<algorithm>.plot -h```

for more information about options and arguments.

For example:

```python3 -m run.dqn.plot dqn-lander```