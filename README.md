# Reinforcement-Learning

## Project Overview
This project implements policy gradient methods in reinforcement learning to train agents in a simulated environment. The goal is to maximize efficiency in navigating and interacting with the environment.

## Installation

Install the required libraries using the following commands:

```bash
pip3 install torch gym-minigrid sensorimotor-checker
```

## Usage
Execute the Python scripts to train and evaluate the reinforcement learning models. Make sure all dependencies are installed prior to running the scripts.

## Code Structure

- **Setup**: Initialization of the environment and required libraries.
- **Model**: Definition of the actor-critic models for policy learning.
- **Training**: Scripts for training the models using reinforcement learning algorithms.
- **Evaluation**: Scripts for evaluating the model's performance within the environment.

## Results and Observations

### Policy Gradients - REINFORCE

![Policy Gradients - REINFORCE](https://github.com/chikap421/Reinforcement-Learning/blob/main/question%206b.png)

The plot above shows the results of the REINFORCE algorithm. It illustrates the reward per episode and the smoothed reward over time. We observe the agent's performance improving as the number of frames increases.

### Vanilla Policy Gradients

![Vanilla Policy Gradients](<https://github.com/chikap421/Reinforcement-Learning/blob/main/question%206c.png>)

The second plot presents the performance of Vanilla Policy Gradients. Similar to the first plot, it displays the reward per episode along with a smoothed version, indicating learning stability and improvement over frames.

## Acknowledgements

Credits to the `gym-minigrid` and `sensorimotor-checker` libraries which were instrumental for this project, and a thank you to all the contributors who have helped in the development of this project.

