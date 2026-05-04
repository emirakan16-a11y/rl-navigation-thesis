"""
Curriculum Manager for dynamic hyperparameter and reward schedule adjustments.
Handles performance-triggered stage transitions with per-layout-family tracking.
"""

import numpy as np
from collections import deque, defaultdict


class CurriculumManager:
    """
    Manages curriculum learning stages with performance-based triggers.
    Supports per-layout-family trigger mode for robust curriculum learning.
    """
    
    def __init__(self, hyperparam_schedule=None, reward_schedule=None, 
                 episodes_per_layout=None, verbose=True):
        """
        Args:
            hyperparam_schedule: List of hyperparameter stage dicts
            reward_schedule: List of reward scale stage dicts
            episodes_per_layout: List of episodes per layout (for per-family mode)
            verbose: Print stage transitions
        """
        self.hyperparam_schedule = hyperparam_schedule or []
        self.reward_schedule = reward_schedule or []
        self.episodes_per_layout = episodes_per_layout or []
        self.verbose = verbose
        
        # Current stages
        self.current_hyperparam_stage = 0
        self.current_reward_stage = 0
        
        # Performance tracking - global
        self.episode_history = {
            'hallway_reached': deque(),
            'terminal_reached': deque(),
            'episode_numbers': deque()
        }
        
        # Performance tracking - per layout
        self.layout_history = defaultdict(lambda: {
            'hallway_reached': deque(),
            'terminal_reached': deque(),
            'episode_numbers': deque()
        })
        
        self.total_episodes = 0
        self.current_layout = None
        
        # Calculate family windows from episodes_per_layout
        self.family_windows = self._calculate_family_windows()
    
    def _calculate_family_windows(self):
        """Calculate window sizes for each layout family from episodes_per_layout"""
        if not self.episodes_per_layout or len(self.episodes_per_layout) < 2:
            return {}
        
        families = {}
        # Assumes pairs: [1a, 1b, 2a, 2b, 3a, 3b]
        for i in range(0, len(self.episodes_per_layout), 2):
            if i + 1 < len(self.episodes_per_layout):
                family_idx = i // 2
                families[family_idx] = self.episodes_per_layout[i] + self.episodes_per_layout[i + 1]
        
        if self.verbose and families:
            print(f"\nLayout Family Windows:")
            for family_idx, window in families.items():
                print(f"  Family {family_idx + 1}: {window} episodes")
        
        return families
    
    def log_episode(self, hallway_reached, terminal_reached, current_layout=None):
        """Log episode outcome for performance tracking"""
        self.total_episodes += 1
        
        # Global tracking
        self.episode_history['hallway_reached'].append(int(hallway_reached))
        self.episode_history['terminal_reached'].append(int(terminal_reached))
        self.episode_history['episode_numbers'].append(self.total_episodes)
        
        # Per-layout tracking
        if current_layout:
            self.current_layout = current_layout
            self.layout_history[current_layout]['hallway_reached'].append(int(hallway_reached))
            self.layout_history[current_layout]['terminal_reached'].append(int(terminal_reached))
            self.layout_history[current_layout]['episode_numbers'].append(self.total_episodes)
        
    def calculate_success_rate(self, metric, window):
        """
        Calculate success rate for given metric (global).
        
        Args:
            metric: 'hallway' or 'terminal'
            window: Number of episodes (None/-1 for all episodes)
        
        Returns:
            Success rate (0.0 to 1.0)
        """
        if metric == 'hallway':
            data = list(self.episode_history['hallway_reached'])
        elif metric == 'terminal':
            data = list(self.episode_history['terminal_reached'])
        else:
            return 0.0
        
        if not data:
            return 0.0
        
        # Use all episodes or sliding window
        if window is None or window == -1:
            subset = data
        else:
            subset = data[-window:]
        
        if not subset:
            return 0.0
            
        return sum(subset) / len(subset)
    
    def calculate_family_success_rate(self, family_idx, metric):
        """
        Calculate success rate for a specific layout family.
        
        Args:
            family_idx: Family index (0=Layout1, 1=Layout2, 2=Layout3)
            metric: 'hallway' or 'terminal'
        
        Returns:
            Success rate (0.0 to 1.0) for that family
        """
        # Get layout names for this family (e.g., Layout1a, Layout1b)
        layout_names = [f'RobustNoRenderLayout{family_idx + 1}a', f'RobustNoRenderLayout{family_idx + 1}b']
        
        combined_data = []
        for layout_name in layout_names:
            if layout_name in self.layout_history:
                if metric == 'hallway':
                    combined_data.extend(list(self.layout_history[layout_name]['hallway_reached']))
                elif metric == 'terminal':
                    combined_data.extend(list(self.layout_history[layout_name]['terminal_reached']))
        
        if not combined_data:
            return 0.0
        
        # Use family-specific window if available
        window = self.family_windows.get(family_idx, None)
        if window is not None and len(combined_data) > window:
            combined_data = combined_data[-window:]
        
        return sum(combined_data) / len(combined_data) if combined_data else 0.0
    
    def check_trigger(self, trigger):
        """
        Check if performance trigger conditions are met.
        
        Args:
            trigger: Dict with 'hallway_success_rate', 'terminal_success_rate', 'mode', 'window', 'require_both'
        
        Returns:
            True if trigger conditions met
        """
        hallway_threshold = trigger.get('hallway_success_rate', 0.0)
        terminal_threshold = trigger.get('terminal_success_rate', 0.0)
        mode = trigger.get('mode', 'global')
        window = trigger.get('window', 50)
        require_both = trigger.get('require_both', True)
        
        if mode == 'per_family':
            # Check each family separately - all must pass
            num_families = len(self.family_windows)
            if num_families == 0:
                # Fallback to global if no families defined
                return self._check_global_trigger(hallway_threshold, terminal_threshold, window, require_both)
            
            for family_idx in range(num_families):
                hallway_rate = self.calculate_family_success_rate(family_idx, 'hallway')
                terminal_rate = self.calculate_family_success_rate(family_idx, 'terminal')
                
                hallway_met = hallway_rate >= hallway_threshold
                terminal_met = terminal_rate >= terminal_threshold
                
                if require_both:
                    family_passed = hallway_met and terminal_met
                else:
                    family_passed = hallway_met or terminal_met
                
                if not family_passed:
                    return False  # One family failed, trigger not met
            
            return True  # All families passed
        
        else:
            # Global mode
            return self._check_global_trigger(hallway_threshold, terminal_threshold, window, require_both)
    
    def _check_global_trigger(self, hallway_threshold, terminal_threshold, window, require_both):
        """Check trigger using global success rates"""
        hallway_rate = self.calculate_success_rate('hallway', window)
        terminal_rate = self.calculate_success_rate('terminal', window)
        
        hallway_met = hallway_rate >= hallway_threshold
        terminal_met = terminal_rate >= terminal_threshold
        
        if require_both:
            return hallway_met and terminal_met
        else:
            return hallway_met or terminal_met
    
    def check_hyperparam_transition(self):
        """
        Check if should transition to next hyperparameter stage.
        
        Returns:
            New stage dict if transition, None otherwise
        """
        if self.current_hyperparam_stage >= len(self.hyperparam_schedule) - 1:
            return None
        
        next_stage_idx = self.current_hyperparam_stage + 1
        next_stage = self.hyperparam_schedule[next_stage_idx]
        
        if 'trigger' not in next_stage:
            return None
        
        if self.check_trigger(next_stage['trigger']):
            self.current_hyperparam_stage = next_stage_idx
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"HYPERPARAMETER STAGE TRANSITION: Stage {next_stage_idx + 1}")
                print(f"Episode: {self.total_episodes}")
                print(f"New hyperparameters: {next_stage['hyperparams']}")
                print(f"{'='*70}\n")
            
            return next_stage
        
        return None
    
    def check_reward_transition(self):
        """
        Check if should transition to next reward stage.
        
        Returns:
            New stage dict if transition, None otherwise
        """
        if self.current_reward_stage >= len(self.reward_schedule) - 1:
            return None
        
        next_stage_idx = self.current_reward_stage + 1
        next_stage = self.reward_schedule[next_stage_idx]
        
        if 'trigger' not in next_stage:
            return None
        
        if self.check_trigger(next_stage['trigger']):
            self.current_reward_stage = next_stage_idx
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"REWARD STAGE TRANSITION: Stage {next_stage_idx + 1}")
                print(f"Episode: {self.total_episodes}")
                print(f"New reward scales: {next_stage['reward_scales']}")
                print(f"{'='*70}\n")
            
            return next_stage
        
        return None
    
    def get_current_hyperparams(self):
        """Get current hyperparameter values"""
        if not self.hyperparam_schedule:
            return {}
        return self.hyperparam_schedule[self.current_hyperparam_stage].get('hyperparams', {})
    
    def get_current_reward_scales(self):
        """Get current reward scale values"""
        if not self.reward_schedule:
            return {}
        return self.reward_schedule[self.current_reward_stage].get('reward_scales', {})
    
    def get_status(self):
        """Get current curriculum status"""
        hallway_rate = self.calculate_success_rate('hallway', 50)
        terminal_rate = self.calculate_success_rate('terminal', 50)
        
        status = {
            'total_episodes': self.total_episodes,
            'hyperparam_stage': self.current_hyperparam_stage + 1,
            'reward_stage': self.current_reward_stage + 1,
            'hallway_success_rate_50': hallway_rate,
            'terminal_success_rate_50': terminal_rate,
            'hallway_success_rate_all': self.calculate_success_rate('hallway', None),
            'terminal_success_rate_all': self.calculate_success_rate('terminal', None),
        }
        
        # Add per-family stats
        for family_idx in range(len(self.family_windows)):
            hallway_family = self.calculate_family_success_rate(family_idx, 'hallway')
            terminal_family = self.calculate_family_success_rate(family_idx, 'terminal')
            status[f'family_{family_idx + 1}_hallway'] = hallway_family
            status[f'family_{family_idx + 1}_terminal'] = terminal_family
        
        return status