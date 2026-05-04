# Reinforcement Learning Navigation Thesis

This repository contains the implementation used in the bachelor thesis at TU Berlin.

## Algorithms

- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- DQN (Deep Q-Network)

## Framework

- Stable-Baselines3
- Gymnasium
- Weights & Biases

## Environment

The experiments were conducted in a custom multi-room navigation environment.

## Experiments

- 300,000 timesteps
- 4 independent runs per algorithm
- identical training conditions

## Sensitivity Analysis

Each algorithm was evaluated using a 2² parameter variation:

- PPO: learning rate, clip range  
- A2C: learning rate, n_steps  
- DQN: exploration fraction, target update interval  

## Notes

This repository documents the experimental setup and implementation used in the thesis.
