"""
Wrapper that enables multiple active terminal positions simultaneously.
Agent can reach ANY of the terminals to succeed.
"""

import gymnasium as gym
import numpy as np


class MultiTerminalWrapper(gym.Wrapper):
    """
    Wrapper that checks distance to ALL terminal positions.
    Success if agent reaches ANY of them.
    """

    def __init__(self, env, terminal_positions):
        """
        Args:
            env: Base environment
            terminal_positions: List of (x, z) tuples for terminal locations
        """
        super().__init__(env)
        self.terminal_positions = terminal_positions
        self.terminal_threshold = 0.3  # Distance threshold for reaching terminal

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check distance to ALL terminals
        agent_pos = self.env.rl_agent_pos if hasattr(self.env, 'rl_agent_pos') else self.env.agent.pos

        min_distance = float('inf')
        closest_terminal_idx = -1

        for idx, (term_x, term_z) in enumerate(self.terminal_positions):
            dx = term_x - agent_pos[0]
            dz = term_z - agent_pos[2]
            distance = np.sqrt(dx**2 + dz**2)

            if distance < min_distance:
                min_distance = distance
                closest_terminal_idx = idx

        # If within threshold of ANY terminal, mark as success
        if min_distance < self.terminal_threshold:
            terminated = True
            info['reached_terminal'] = True
            info['terminal_reached'] = closest_terminal_idx
            info['distance_to_terminal'] = min_distance

            # Add terminal reward if not already given
            if not info.get('original_terminated', False):
                reward += 100  # Terminal reward bonus

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment - don't change terminal location"""
        obs, info = self.env.reset(**kwargs)

        # Make both terminals "active" by not changing env.terminal_location
        # This way the agent can navigate to either one

        return obs, info
