"""
Configuration for training with curriculum learning.
"""

import time
import os
from typing import Optional, List


class TrainingConfig:
    """Configuration class for training parameters with curriculum learning"""
    
    def __init__(self):
        # Path for saving logs to external SSD
        self.log_base_path = "XXX" #this needs your own path!!!!
        
        # Generate random seed based on current time
        self.seed = int(time.time()) % 10000
        
        # Run counter for unique naming
        self._run_counter = self._get_next_run_counter()
        
        # Algorithm selection
        self.algorithm = "PPO"  # Options: "PPO", "A2C", "DQN"
        
        # Environment parameters
        self.max_episode_steps = 500
        self.steps_until_hallway = 500
        
        # Training parameters
        self.timesteps_per_iteration = 12000 #that was the most appropriate one for the SARL using PPO - you might want to change this
        self.num_iterations = 100 #the total length - good results, if environment very randomn, maybe with set to 1000
        self.total_timesteps = self.timesteps_per_iteration * self.num_iterations
        
        # Curriculum parameters
        self.enable_terminal_randomization = True
        self.enable_layout_switching = False
        self.terminal_change_interval = 1 # terminal location changes every n episoes
        self.layout_change_interval = 0 # layout changes every n episodes
        self.episodes_per_layout = None #[200, 200, 200, 200, 200, 200]  # Episodes for each layout (pairs = families)
        
        # Number of parallel environments
        self.num_envs = 1 # for training (you should use 1)
        
        # Base reward scales (can be overridden in sweep configs)
        self.reward_scales = {
            'reward_orientation_scale': 1.0,
            'reward_distance_scale': 0.0,
            'punishment_distance_scale': 0.0,
            'penalty_stagnation_scale': 1.0,
            'punishment_time_scale': 0.0,
            'reward_hallway_scale': 1.0,
            'reward_connection_scale': 0.0,
            'reward_terminal_scale': 1.0,
            'punishment_terminal_scale': 0.0,
            'punishment_room_scale': 0.0,
            'wall_collision_scale': 1.0
        }
        
        # -------------------------------------------------
        # HIER ÄNDERN!
        # Common hyperparameters (shared across algorithms)
        self.learning_rate = 0.0004
        self.gamma = 0.99
        self.ent_coef = 0.01
        
        # Algorithm-specific base hyperparameters
        self.algorithm_kwargs = {
            'PPO': {
                'policy_type': "MlpPolicy",
                'n_epochs': 10,
                'clip_range': 0.2,
                'gae_lambda': 0.99,
                'n_steps': 2048,
                'batch_size': 64,
            },
            'A2C': {
                'policy_type': "MlpPolicy",
                'n_steps': 20,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
            },
            'DQN': {
                'policy_type': "MlpPolicy",
                'buffer_size': 100000,
                'learning_starts': 1000,
                'target_update_interval': 1000,
                'exploration_fraction': 0.3,   # 30% of training with decaying epsilon
                'exploration_initial_eps': 1.0, # Start with 100% random
                'exploration_final_eps': 0.05,  # End with 5% random
            }
        }
        # -------------------------------------------------

        # Logging parameters
        self.project_name = "simple_multi-single" # this needs to be the exact name of the wandb project you created and want to use
        self.run_name = f"run_{self._run_counter}_{self.algorithm}_seed_{self.seed}" # this is the naming convention for the runs
    
    def get_hyperparams(self):
        """Get common hyperparameters"""
        return {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'ent_coef': self.ent_coef
        }
    
    def get_initial_reward_scales(self):
        """Get initial reward scales"""
        return self.reward_scales

    def override_reward_scale(self, scale_name, value):
        """
        Override a specific reward scale.
        Useful for sweep experiments.

        Args:
            scale_name: Name of the reward scale (e.g., 'penalty_stagnation_scale')
            value: New value for the scale
        """
        if scale_name in self.reward_scales:
            self.reward_scales[scale_name] = value
        else:
            raise KeyError(f"Reward scale '{scale_name}' not found in config")

    def _get_next_run_counter(self):
        """Return incrementing counter for run naming (1,2,3,...)."""
        counter_file = os.path.join(os.path.dirname(__file__), ".run_counter")
        counter = 0
        if os.path.exists(counter_file):
            try:
                with open(counter_file, "r") as f:
                    counter = int(f.read().strip())
            except (ValueError, OSError):
                counter = 0
        next_counter = counter + 1
        try:
            with open(counter_file, "w") as f:
                f.write(str(next_counter))
        except OSError:
            # If writing fails, still return computed counter
            pass
        return next_counter

    def override_hyperparam(self, param_name, value):
        """
        Override a specific hyperparameter.
        Useful for sweep experiments.

        Args:
            param_name: Name of the hyperparameter (e.g., 'learning_rate', 'ent_coef', 'gamma')
            value: New value for the parameter
        """
        if hasattr(self, param_name):
            setattr(self, param_name, value)
        else:
            raise KeyError(f"Hyperparameter '{param_name}' not found in config")

    def override_algorithm_kwarg(self, param_name, value):
        """
        Override algorithm-specific parameter.

        Args:
            param_name: Name of the parameter (e.g., 'n_steps', 'clip_range')
            value: New value for the parameter
        """
        if self.algorithm in self.algorithm_kwargs:
            if param_name in self.algorithm_kwargs[self.algorithm]:
                self.algorithm_kwargs[self.algorithm][param_name] = value
            else:
                raise KeyError(f"Parameter '{param_name}' not found for algorithm '{self.algorithm}'")
        else:
            raise KeyError(f"Algorithm '{self.algorithm}' not found in config")

    def to_dict(self):
        """Convert config to dictionary for wandb"""
        return {
            'seed': self.seed,
            'algorithm': self.algorithm,
            'log_base_path': self.log_base_path,
            'max_episode_steps': self.max_episode_steps,
            'steps_until_hallway': self.steps_until_hallway,
            'timesteps': self.timesteps_per_iteration,
            'iterations': self.num_iterations,
            'total_timesteps': self.total_timesteps,
            'enable_terminal_randomization': self.enable_terminal_randomization,
            'enable_layout_switching': self.enable_layout_switching,
            'terminal_change_interval': self.terminal_change_interval if self.enable_terminal_randomization else 0,
            'layout_change_interval': self.layout_change_interval if self.enable_layout_switching else 0,
            'episodes_per_layout': self.episodes_per_layout,
            'num_envs': self.num_envs,
            **self.get_hyperparams(),
            **{f'reward_{k}': v for k, v in self.get_initial_reward_scales().items()}
        }
    
    def print_config(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        print(f"Algorithm: {self.algorithm}")
        print(f"Log Path: {self.log_base_path}")
        print(f"Seed: {self.seed}")
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Iterations: {self.num_iterations}")
        print(f"\nHyperparameters:")
        for key, value in self.get_hyperparams().items():
            print(f"  {key}: {value}")
        print(f"\nAlgorithm-specific kwargs:")
        for key, value in self.algorithm_kwargs.get(self.algorithm, {}).items():
            print(f"  {key}: {value}")
        print(f"\nReward Scales:")
        for key, value in self.get_initial_reward_scales().items():
            if value != 0.0:
                print(f"  {key}: {value}")
        print("="*60 + "\n")