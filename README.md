# Reinforcement Learning with Policy Gradients

## Project Overview
This project focuses on implementing and experimenting with policy gradient methods in reinforcement learning, specifically targeting the complex challenges presented by the `gym-minigrid` DoorKey environment. Our approach utilizes advanced actor-critic models to efficiently navigate and solve spatial reasoning tasks in a simulated grid environment. The primary goal is to develop an agent capable of maximizing efficiency in environment interaction, leveraging visual perception and sequential decision-making to achieve success.

## Installation

To set up the necessary environment for running the experiments, please install the required libraries using the following command:

```bash
pip3 install torch gym-minigrid sensorimotor-checker
```
This command ensures all dependencies, including PyTorch for model development and gym-minigrid for the simulation environment, are correctly installed.

## Usage

After installing the dependencies, execute the Python scripts to begin training and evaluating the reinforcement learning models. These scripts encompass the entire workflow from initialization and training to evaluation of the agents within the specified environment.

## Code Structure

- **Environment Setup**: Custom initialization of the `DoorKeyEnv5x5` environment from `gym_minigrid`, designed to challenge the agent with tasks involving key-finding and door-opening using visual inputs.
- **Model Development**: Implementation of a sophisticated actor-critic model (`ACModel`) using convolutional neural networks (CNNs) to process visual inputs from the environment. This model structure is pivotal for learning efficient policies through direct interaction.
- **Training Framework**: Comprehensive scripts for training the models employing policy gradient methods, with detailed configurations encapsulated in a `Config` class for hyperparameters and training settings.
- **Evaluation Mechanics**: Dedicated scripts for assessing the trained model's performance, focusing on its ability to navigate and solve the presented tasks within the environment effectively.

## Results and Observations

### Understanding through Visual Analysis

- **Policy Gradients - REINFORCE**: The provided plot illustrates the agent's learning progress over episodes, showing a clear trend of performance improvement. This demonstrates the agent's growing proficiency in environment navigation and task completion.
  
  ![Policy Gradients - REINFORCE](https://github.com/chikap421/Reinforcement-Learning/blob/main/question%206b.png)

- **Vanilla Policy Gradients**: Similar to the REINFORCE results, this plot highlights the effectiveness of Vanilla Policy Gradients in stabilizing and enhancing the learning process, as evidenced by the smooth increase in rewards over time.
  
  ![Vanilla Policy Gradients](https://github.com/chikap421/Reinforcement-Learning/blob/main/question%206c.png)

## Acknowledgements

This project owes its success to the foundational `gym-minigrid` and `sensorimotor-checker` libraries, which provided the essential environment and tools for our reinforcement learning experiments. Special thanks to the PyTorch community for the robust deep learning framework that underpins our models. We also extend our gratitude to all contributors and collaborators whose insights and efforts have significantly propelled this project forward.

