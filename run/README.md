# Reinforcement Learning Algorithms Experiments

## Introduction

Here is example code to run the algorithms on OpenAI Gym environments. Also
includes experiments on environments to show performance.

## Examples

Example experiments (training code and parameters, trained models, logs and plots from training) 
included are:

* Vanilla Policy Gradients (VPG)
    * CartPole-v1
    * RoboschoolInvertedPendulum-v1
    * LunarLanderContinuous-v2
    * RoboschoolHalfCheetah-v1

* Deep Q-Networks (DQN)
    * PongNoFrameskip-v4 (note to run the Pong model, need to first unzip it)
    * LunarLander-v2
    * (on /snake branch) [snake-grid-v0](https://github.com/olliejday/snake-rl)

* Advantage Actor Critic (A2C)
    * RoboschoolInvertedPendulum-v1
    * CartPole-v1
    * RoboschoolHalfCheetah-v1
    
* Soft Actor Critic (SAC)
    * RoboschoolInvertedPendulum-v1
    * LunarLanderContinuous-v2
    * RoboschoolHalfCheetah-v1
    * RoboschoolAnt-v1
    
    
All experiments (except some that are highly expensive to run), are run over three random
seeds. Plots then show mean and standard deviation across seeds to give a more reliable
indicator of the algorithms' behaviour.
    
For some of the environments that multiple algorithms were run on, we include plots 
to compare the training and performance of the different algorithms. These are included
in the /run/compare directory. They use the returns for episodes gathered during training
so are not the most principled evaluation, but do give a good idea of the speed and 
stability.

## Usage

For each algorithm we have a template for training, running and plotting interface.

All examples are OpenAI Gym environments, but some included are Roboschool and Atari environments
which require installing too.

## Training

Train model by setting up a training function as seen in examples in `run/<algorithm>/train.py`. You need to 
provide hyperparamers, environment name etc.

Then add this custom training function, `train_fn` to the options dictionary in `train.py`. You must 
give it a name `<experiment>`. Add a line:

```options['<experiment>'] = <train_fn>```  

Then to train this custom training setup, call from the terminal:

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

```python3 -m run.<algorithm>.plot <experiment_dir>```

Where:
 
`<algorithm>` is the algorithm<br/>
`<experiment_dir>` is the name of the experiment directory (ie. the top directory storing logs and models) eg. `dqn-lander`<br/>


Run 

```python3 -m run.<algorithm>.plot -h```

for more information about options and arguments.

eg. ```python3 -m run.dqn.plot dqn-lander```