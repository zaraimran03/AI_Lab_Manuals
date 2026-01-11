# Lab 12: Implementation of Reinforcement Learning

## Lab Overview
Implementation of Reinforcement Learning using Gymnasium environments (CartPole, MountainCar).

## Objectives
- Understand RL concepts (agent, environment, state, action, reward)
- Use Gymnasium environments
- Visualize agent behavior with Pygame
- Compare random vs rule-based policies

## Tasks Completed

### Experiment 1: CartPole Environment
- Created CartPole-v1 environment
- Implemented random action policy
- Displayed score on screen using Pygame
- Ran 20 episodes
- Analyzed score variations
- Modified to show episode number
- Printed maximum score
- Tested rule-based action (pole angle)

### Experiment 2: MountainCar Environment
- Created MountainCar-v0 environment
- Implemented velocity-based policy
- Built momentum to reach goal
- Compared with random actions
- Analyzed reward mechanism
- Explained terminated vs truncated
- Discussed importance of momentum

### Lab Questions Answered
- Identified agent, environment, state, action, reward
- Explained env.reset() return values
- Differentiated terminated vs truncated
- Analyzed score calculation
- Discussed random vs intelligent agents
- Explained Pygame visualization
- Compared CartPole with MountainCar

## Libraries Used
```python
import gymnasium as gym
import pygame
import numpy as np
```

## Key Learnings
- RL interaction loop
- Exploration vs exploitation
- Delayed rewards problem
- Momentum in control tasks
- State representation
- Episode termination conditions
- Rule-based policies vs learning

