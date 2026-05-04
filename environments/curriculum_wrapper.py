"""
Environment wrapper that handles:
1. Terminal location randomization (every N episodes)
2. Layout switching (every M episodes)
3. Seamless transitions while maintaining learning continuity
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.monitor import Monitor
import random


class CurriculumEnvWrapper(gym.Wrapper):
    """
    Wrapper that manages terminal randomization and layout switching.
    
    Args:
        env_class: The environment class to use (will be instantiated internally)
        layout_classes: List of layout classes to cycle through
        max_episode_steps: Maximum steps per episode
        steps_until_hallway: Steps until hallway timeout
        reward_scales: Reward scaling dictionary
        terminal_change_interval: Change terminal location every N episodes (0 to disable)
        layout_change_interval: Change layout every M episodes (0 to disable)
        episodes_per_layout: List of episodes for each layout (None = use layout_change_interval)
        seed: Random seed
    """
    
    def __init__(self, 
                 layout_classes,
                 max_episode_steps,
                 steps_until_hallway,
                 reward_scales=None,
                 terminal_change_interval=20,
                 layout_change_interval=100,
                 episodes_per_layout=None,
                 seed=None):
        
        self.layout_classes = layout_classes
        self.max_episode_steps = max_episode_steps
        self.steps_until_hallway = steps_until_hallway
        self.reward_scales = reward_scales
        self.terminal_change_interval = terminal_change_interval
        self.layout_change_interval = layout_change_interval
        self.episodes_per_layout = episodes_per_layout
        self.seed_value = seed

        # Validate episodes_per_layout if provided
        if self.episodes_per_layout is not None:
            if len(self.episodes_per_layout) != len(self.layout_classes):
                raise ValueError(
                    f"episodes_per_layout length ({len(self.episodes_per_layout)}) "
                    f"must match layout_classes length ({len(self.layout_classes)})"
                )
        elif len(self.layout_classes) == 1:
            # Single layout mode: disable layout switching
            self.layout_change_interval = 0
        
        # Episode counters
        self.episode_count = 0
        self.episodes_since_terminal_change = 0
        self.episodes_since_layout_change = 0
        
        # Current layout index
        self.current_layout_idx = 0
        
        # Initialize with first layout
        self._create_new_env(self.current_layout_idx)
        
        super().__init__(self.env)
        
        # Store observation and action spaces (same across all layouts)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Set initial terminal location
        self._randomize_terminal()
        
        print(f"Initialized CurriculumEnvWrapper:")
        print(f"  - Number of layouts: {len(self.layout_classes)}")
        print(f"  - Terminal change interval: {terminal_change_interval} episodes (disabled if 0)")
        print(f"  - Layout change interval: {layout_change_interval} episodes (disabled if 0)")
        print(f"  - Starting with layout: {self.layout_classes[0].__name__}")

        print(f"\nCurriculum Learning Configuration:")
        print(f"  - Number of layouts: {len(self.layout_classes)}")
        
        if self.episodes_per_layout:
            print(f"  - Episodes per layout: {self.episodes_per_layout}")
            print(f"  - Total episodes: {sum(self.episodes_per_layout)}")
        else:
            print(f"  - Episodes per layout: {layout_change_interval} (uniform)")
        
        print(f"  - Terminal change interval: {terminal_change_interval} episodes (disabled if 0)")
        print(f"  - Starting with layout: {self.layout_classes[0].__name__}")
    
    def _create_new_env(self, layout_idx):
        """Create a new environment instance with the specified layout"""
        layout_class = self.layout_classes[layout_idx]
        
        # Close old environment if it exists
        if 'env' in self.__dict__ and self.env is not None:
            try:
                self.env.close()
            except:
                pass
        
        # Create new environment
        self.env = layout_class(
            max_episode_steps=self.max_episode_steps,
            steps_until_hallway=self.steps_until_hallway,
            reward_scales=self.reward_scales
        )
        
        # Set seed if provided
        if self.seed_value is not None:
            self.env.seed(self.seed_value)
        
        print(f"Switched to layout: {layout_class.__name__}")
    
    def _randomize_terminal(self):
        """Randomize the terminal location"""
        self.env.set_terminal_location()
        terminal_pos = self.env.terminal_location
        #print(f"Terminal location set to: ({terminal_pos[0]:.2f}, {terminal_pos[1]:.2f})")
    
    def _should_change_terminal(self):
        """Check if terminal location should be changed"""
        if self.terminal_change_interval <= 0:
            return False
        return self.episodes_since_terminal_change >= self.terminal_change_interval
    
    def _should_change_layout(self):
        """Check if it's time to change the layout"""
        if self.layout_change_interval == 0:
            return False

        if self.episodes_per_layout:
            episodes_needed = self.episodes_per_layout[self.current_layout_idx]
        else:
            episodes_needed = self.layout_change_interval
        
        return self.episodes_since_layout_change >= episodes_needed
    
    def reset(self, **kwargs):
        """Reset the environment, potentially changing terminal or layout"""
        
        # Check if we should change layout (less frequent)
        if self._should_change_layout():
            # Cycle to next layout
            self.current_layout_idx = (self.current_layout_idx + 1) % len(self.layout_classes)
            self._create_new_env(self.current_layout_idx)
            self.episodes_since_layout_change = 0
            self.episodes_since_terminal_change = 0  # Also reset terminal counter
            
            # Set new terminal location for new layout
            self._randomize_terminal()
        
        # Check if we should change terminal (more frequent)
        elif self._should_change_terminal():
            self._randomize_terminal()
            self.episodes_since_terminal_change = 0
        
        # Reset the environment
        observation, info = self.env.reset(**kwargs)
        
        # Increment counters
        self.episode_count += 1
        self.episodes_since_terminal_change += 1
        self.episodes_since_layout_change += 1
        
        # Add curriculum info to the info dict
        info['curriculum'] = {
            'total_episodes': self.episode_count,
            'current_layout': self.layout_classes[self.current_layout_idx].__name__,
            'episodes_since_terminal_change': self.episodes_since_terminal_change,
            'episodes_since_layout_change': self.episodes_since_layout_change,
            'terminal_location': self.env.terminal_location
        }
        
        return observation, info
    
    def step(self, action):
        """Forward the step to the underlying environment"""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Add curriculum info
        info['curriculum'] = {
            'total_episodes': self.episode_count,
            'current_layout': self.layout_classes[self.current_layout_idx].__name__,
            'terminal_location': self.env.terminal_location
        }
        
        return observation, reward, terminated, truncated, info
    
    def seed(self, seed=None):
        """Set the seed for the wrapper and underlying environment"""
        self.seed_value = seed
        if 'env' in self.__dict__ and self.env is not None:
            return self.env.seed(seed)
        return [seed]
    
    def close(self):
        """Close the underlying environment"""
        if 'env' in self.__dict__ and self.env is not None:
            self.env.close()


def make_curriculum_env(layout_classes,
                        max_episode_steps,
                        steps_until_hallway,
                        reward_scales=None,
                        terminal_change_interval=20,
                        layout_change_interval=100,
                        episodes_per_layout=None,
                        seed=None):
    """
    Factory function to create a curriculum environment.
    
    Args:
        layout_classes: List of layout classes to use
        max_episode_steps: Maximum steps per episode
        steps_until_hallway: Steps until hallway timeout
        reward_scales: Reward scaling dictionary
        terminal_change_interval: Change terminal every N episodes (0 to disable)
        layout_change_interval: Change layout every M episodes (0 to disable)
        episodes_per_layout: List of episodes for each layout (None = uniform)
        seed: Random seed
        
    Returns:
        Wrapped and monitored environment
    """
    def _init():
        try:
            env = CurriculumEnvWrapper(
                layout_classes=layout_classes,
                max_episode_steps=max_episode_steps,
                steps_until_hallway=steps_until_hallway,
                reward_scales=reward_scales,
                terminal_change_interval=terminal_change_interval,
                layout_change_interval=layout_change_interval,
                episodes_per_layout=episodes_per_layout,  # NEW
                seed=seed
            )
            env = Monitor(env)
            return env
        except Exception as e:
            print(f"Error initializing curriculum environment with seed {seed}:")
            import traceback
            print(traceback.format_exc())
            raise
    
    return _init
