# Reinforcement Learning for Navigation in Multi-Room Environments

This repository contains the implementation used in a bachelor thesis conducted at TU Berlin.  
The work focuses on comparing different reinforcement learning algorithms in a navigation task within a custom multi-room environment.

## Algorithms

The following algorithms were implemented and evaluated:

- PPO (Proximal Policy Optimization)  
- A2C (Advantage Actor-Critic)  
- DQN (Deep Q-Network)  

## Framework & Tools

- Stable-Baselines3  
- Gymnasium  
- PyTorch  
- Weights & Biases (for experiment tracking and logging)  

## Environment

The experiments were conducted in a custom multi-room navigation environment.  
The agent’s objective is to reach a dynamically changing target terminal within a fixed number of steps.

## Experimental Setup

- Training budget: 300,000 timesteps  
- 4 independent runs (different seeds) per algorithm  
- Identical training conditions for fair comparison  
- Metrics recorded via Weights & Biases and exported as CSV  

## Sensitivity Analysis

A 2² experimental design was used to evaluate the impact of key hyperparameters:

- PPO: learning rate, clip range  
- A2C: learning rate, n_steps  
- DQN: exploration fraction, target update interval  

Each parameter was tested with two levels (Low / High), resulting in four configurations (LL, LH, HL, HH).

## Usage

To run baseline experiments:
python run_all_experiments.py

To run sensitivity analysis:
python run_sensitivity.py


## Reproducibility Notes

Experiments were executed sequentially due to computational constraints.  
Baseline results are reported as aggregated values (mean over multiple runs).  
Sensitivity analysis was conducted using a single seed per configuration.

## Requirements

Install dependencies with:

pip install -r requirements.txt


## Disclaimer

This repository documents the implementation and experimental setup used in the thesis.  
All analysis, results, and interpretations are presented in the written thesis document.
