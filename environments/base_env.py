"""
Base environment class for the escape room RL task.
All environment variants should inherit from this class.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from miniworld.entity import Agent, Box, COLORS
from miniworld.miniworld import MiniWorldEnv, Room
import random


class BaseEscapeEnv(MiniWorldEnv):
    """
    Base class for escape room environments.
    Defines common functionality and interface that all variants must implement.
    """
    
    def __init__(self, max_episode_steps, steps_until_hallway, reward_scales=None, **kwargs):
        # World dimensions - can be overridden by child classes BEFORE calling super().__init__()
        # Only set defaults if not already set by child class
        if not hasattr(self, 'world_width'):
            self.world_width = 8.0
        if not hasattr(self, 'world_depth'):
            self.world_depth = 20.0
        
        # Default reward scales
        self.reward_scales = reward_scales or {
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
        
        # Episode parameters
        self.max_episode_steps = max_episode_steps
        self.steps_until_hallway = steps_until_hallway
        
        # Tracking variables
        self.has_reached_hallway = False
        self.hallway_reward_given = False
        self.started_in_hallway = False
        self.non_hallway_steps = 0
        self.stagnant_steps = 0
        self.reached_terminal_area = False
        self.placement_incrementer = 0
        
        # Terminal location - will be set in child classes or randomized
        self.terminal_location = None
        self.terminal_zone = None  # For orientation-aware zone detection
        
        super().__init__(max_episode_steps=max_episode_steps, **kwargs)
        
        # Initialize room and entity lists
        self.rooms = []
        self.boxes = []
        self.previous_room = None
        
        # Define observation space (shared across all variants)
        self.observation_space = spaces.Box(
            low=np.array([
                0, 0,                   # normalized position of agent
                0, 0,                   # normalized position of terminal
                0,                      # normalized distance to terminal
                -1, -1,                 # normalized agents direction vector
                -1, -1,                 # normalized direction to terminal 
                -1,                     # direction difference
                0, 0, 0, 0, 0,          # lidar measurements (right, left, front, FR, FL)
                0,                      # normalized step-count
                0,                      # normalized stagnation counter
                0,                      # room category
            ], dtype=np.float32),
            high=np.array([
                1, 1,                   # normalized position of agent
                1, 1,                   # normalized position of terminal
                1,                      # normalized distance to terminal
                1, 1,                   # normalized agents direction vector
                1, 1,                   # normalized direction to terminal 
                1,                      # direction difference
                1, 1, 1, 1, 1,          # lidar measurements (right, left, front, FR, FL)
                1,                      # normalized step-count
                1,                      # normalized stagnation counter
                4,                      # room category
            ], dtype=np.float32)
        )
        
        # Action space: right, left, forward (shared across all variants)
        self.action_space = spaces.Discrete(3)
    
    def _gen_world(self):
        """
        Generate the world. Must be implemented by child classes.
        Should create rooms, place boxes, and set up connections.
        """
        raise NotImplementedError("Child classes must implement _gen_world()")
    
    def _add_boxes(self):
        """
        Add boxes as obstacles. Can be overridden by child classes.
        """
        raise NotImplementedError("Child classes must implement _add_boxes()")
    
    def _get_hallway_terminal_positions(self):
        """
        Get valid terminal positions along the hallway.
        Returns list of (x, z) tuples.
        Should be implemented by child classes based on their hallway layout.
        """
        raise NotImplementedError("Child classes must implement _get_hallway_terminal_positions()")
    
    def set_terminal_location(self, location=None):
        """
        Set or randomize terminal location.
        If location is None, randomly selects from valid hallway positions.
        """
        if location is None:
            valid_positions = self._get_hallway_terminal_positions()
            self.terminal_location = list(random.choice(valid_positions))
        else:
            self.terminal_location = list(location)
        
        # Generate terminal zone for orientation-aware detection
        self._generate_terminal_zone()
    
    def _detect_door_orientation(self, x, z):
        """
        Detect if door is on x-axis edge or z-axis edge.
        Returns: 'x-axis', 'z-axis', or 'unknown'
        """
        x_tolerance = 0.5
        z_tolerance = 0.5
        
        # Check if door is on left or right edge (x-axis)
        if abs(x - 0.0) < x_tolerance or abs(x - self.world_width) < x_tolerance:
            return 'z-axis'  # Door on side edge → terminal zone extends along z-axis
        
        # Check if door is on top or bottom edge (z-axis)
        if abs(z - 0.0) < z_tolerance or abs(z - self.world_depth) < z_tolerance:
            return 'x-axis'  # Door on top/bottom edge → terminal zone extends along x-axis
        
        return 'unknown'
    
    def _generate_terminal_zone(self):
        """
        Generate terminal zone (area) based on door orientation.
        Creates min/max bounds for zone detection.
        """
        if self.terminal_location is None:
            return
        
        x, z = self.terminal_location[0], self.terminal_location[1]
        door_orientation = self._detect_door_orientation(x, z)
        
        # Zone dimensions
        extension = 0.4  # How far into the hallway
        side_tolerance = 0.5  # ±0.5 units on the sides
        
        if door_orientation == 'z-axis':  # Door on x-axis edge
            # Zone extends in z-direction
            if abs(x - 0.0) < 0.5:  # Left edge
                self.terminal_zone = {
                    'x_min': 0.0,
                    'x_max': extension,
                    'z_min': z - side_tolerance,
                    'z_max': z + side_tolerance
                }
            else:  # Right edge
                self.terminal_zone = {
                    'x_min': self.world_width - extension,
                    'x_max': self.world_width,
                    'z_min': z - side_tolerance,
                    'z_max': z + side_tolerance
                }
        
        elif door_orientation == 'x-axis':  # Door on z-axis edge
            # Zone extends in x-direction
            if abs(z - 0.0) < 0.5:  # Bottom edge
                self.terminal_zone = {
                    'x_min': x - side_tolerance,
                    'x_max': x + side_tolerance,
                    'z_min': 0.0,
                    'z_max': extension
                }
            else:  # Top edge
                self.terminal_zone = {
                    'x_min': x - side_tolerance,
                    'x_max': x + side_tolerance,
                    'z_min': self.world_depth - extension,
                    'z_max': self.world_depth
                }
        else:
            # Fallback to point-based detection if orientation unknown
            self.terminal_zone = None
    
    def _is_agent_in_terminal_zone(self, agent_pos):
        """
        Check if agent nose is within terminal zone.
        Returns True if agent is in the zone, False otherwise.
        """
        if self.terminal_zone is None:
            # Fallback to distance-based detection
            distance = np.linalg.norm(
                np.array([agent_pos[0], agent_pos[2]]) - 
                np.array([self.terminal_location[0], self.terminal_location[1]])
            )
            return distance < 0.2
        
        x, z = agent_pos[0], agent_pos[2]
        return (self.terminal_zone['x_min'] <= x <= self.terminal_zone['x_max'] and
                self.terminal_zone['z_min'] <= z <= self.terminal_zone['z_max'])
    
    def generate_connections(self, room_pairs, special_configs):
        """Generate possible connection points between rooms"""
        connections = {}
        
        if isinstance(special_configs, list):
            special_configs = {}
        
        def add_connection(room1, room2, id1, id2, is_vertical=False):
            special_config = special_configs.get((id1, id2)) or special_configs.get((id2, id1))
            
            if is_vertical:
                x_start = round(max(room1.min_x, room2.min_x) + 0.5, 1)
                x_end = round(min(room1.max_x, room2.max_x) - 0.5, 1)
                
                if x_end > x_start:
                    z_top = special_config['z_range'][1] if special_config and 'z_range' in special_config else room1.max_z
                    z_bottom = special_config['z_range'][0] if special_config and 'z_range' in special_config else room2.min_z
                    
                    connections[((x_start, x_end), 
                            (round(z_top - 0.1, 1), z_top))] = {
                        'connection': f'{id1}-{id2}', 'created': False
                    }
                    connections[((x_start, x_end),
                            (z_bottom, round(z_bottom + 0.1, 1)))] = {
                        'connection': f'{id2}-{id1}', 'created': False
                    }
            else:
                z_start = round(max(room1.min_z, room2.min_z) + 0.5, 1)
                z_end = round(min(room1.max_z, room2.max_z) - 0.5, 1)
                
                if special_config and 'z_range' in special_config:
                    z_start, z_end = special_config['z_range']
                
                if z_end > z_start:
                    connections[((round(room1.max_x - 0.1, 1), room1.max_x),
                            (z_start, z_end))] = {
                        'connection': f'{id1}-{id2}', 'created': False
                    }
                    connections[((room2.min_x, round(room2.min_x + 0.1, 1)),
                            (z_start, z_end))] = {
                        'connection': f'{id2}-{id1}', 'created': False
                    }
        
        for room1, room2, id1, id2 in room_pairs:
            if abs(room1.max_x - room2.min_x) < 0.3:
                add_connection(room1, room2, id1, id2)
            elif abs(room2.max_x - room1.min_x) < 0.3:
                add_connection(room2, room1, id2, id1)
                
            if abs(room1.max_z - room2.min_z) < 0.3:
                add_connection(room1, room2, id1, id2, True)
            elif abs(room2.max_z - room1.min_z) < 0.3:
                add_connection(room2, room1, id2, id1, True)
                    
        return connections
    
    def _get_current_room(self):
        """Determine which room the agent is currently in"""
        agent_pos = self.agent.pos
        boundary_tolerance = 0.2
        
        for room in self.rooms:
            if ((room.min_x - boundary_tolerance <= agent_pos[0] <= room.max_x + boundary_tolerance) and 
                (room.min_z - boundary_tolerance <= agent_pos[2] <= room.max_z + boundary_tolerance)):
                
                for attr_name, attr_value in vars(self).items():
                    if attr_name.startswith('room') and attr_value == room:
                        return attr_name
        
        if hasattr(self, 'previous_room'):
            return self.previous_room
                
        return "unknown"
    
    def _get_room_category(self, room_name):
        """
        Get room category for observation.
        Should be consistent across all environment variants.
        0: Hallway (roomD)
        1-3: Other rooms
        4: Unknown
        """
        if room_name == 'roomD':
            return 0
        elif room_name == 'roomA':
            return 1
        elif room_name == 'roomB':
            return 2
        elif room_name == 'roomC':
            return 3
        else:
            return 4
    
    def get_wall_distance(self, pos, direction, max_distance=10):
        """Cast a ray from position in given direction and return distance to nearest wall"""
        step_size = 0.1
        current_pos = np.array(pos)
        
        for i in range(int(max_distance / step_size)):
            current_pos = current_pos + direction * step_size
            
            in_room = False
            for room in self.rooms:
                if (room.min_x <= current_pos[0] <= room.max_x and
                    room.min_z <= current_pos[2] <= room.max_z):
                    in_room = True
                    break
            
            if not in_room:
                return i * step_size
                
        return max_distance
    
    def get_lidar_measurements(self):
        """Get distances to walls in different directions"""
        forward_dir = self.agent.dir_vec
        right_dir = np.array([-forward_dir[2], 0, forward_dir[0]])
        left_dir = np.array([forward_dir[2], 0, -forward_dir[0]])
        
        forward_right_dir = forward_dir + right_dir
        forward_right_dir = forward_right_dir / np.linalg.norm(forward_right_dir)
        
        forward_left_dir = forward_dir + left_dir
        forward_left_dir = forward_left_dir / np.linalg.norm(forward_left_dir)
        
        forward_dist = self.get_wall_distance(self.agent.pos, forward_dir)
        left_dist = self.get_wall_distance(self.agent.pos, left_dir)
        right_dist = self.get_wall_distance(self.agent.pos, right_dir)
        forward_right_dist = self.get_wall_distance(self.agent.pos, forward_right_dir)
        forward_left_dist = self.get_wall_distance(self.agent.pos, forward_left_dir)
        
        return right_dist, left_dist, forward_dist, forward_right_dist, forward_left_dist
    
    def _normalize_terminal_distance(self, distance):
        """Higher number = closer to terminal (better for agent)"""
        max_dist = np.sqrt(self.world_width**2 + self.world_depth**2)
        return (max_dist - distance) / max_dist
    
    def _agent_touches_wall(self, agent_nose_pos):
        """Check if the agent is touching a wall between rooms (for door creation)"""
        touching_threshold = 0.1
        for (x_range, z_range), data in self.connections.items():
            if (x_range[0]-touching_threshold <= agent_nose_pos[0] <= x_range[1]+touching_threshold and 
                z_range[0]-touching_threshold <= agent_nose_pos[2] <= z_range[1]+touching_threshold):
                return (x_range, z_range)
        return None
    
    def _create_doors(self, new_connection):
        """Create a door at a connection point"""
        if not hasattr(self, 'special_connection_groups'):
            self.special_connection_groups = {}

        x_range, z_range = new_connection
        connection = self.connections[new_connection]['connection']
        room1_id, room2_id = connection.split('-')
        
        room1_attr = f"room{room1_id}"
        room2_attr = f"room{room2_id}"
        
        room1 = getattr(self, room1_attr)
        room2 = getattr(self, room2_attr)
        
        door_width = 1.0
        min_offset = 0.05
        
        if abs(room1.max_x - room2.min_x) < 0.3 or abs(room2.max_x - room1.min_x) < 0.3:
            left_room = room1 if room1.max_x < room2.min_x else room2
            right_room = room2 if left_room == room1 else room1
            
            min_z = self.agent.pos[2] - door_width/2
            min_z = np.clip(min_z,
                        max(left_room.min_z, right_room.min_z) + min_offset,
                        min(left_room.max_z, right_room.max_z) - door_width - min_offset)
            
            self.connect_rooms(left_room, right_room, min_z=min_z, max_z=min_z + door_width)
        else:
            top_room = room1 if room1.max_z > room2.max_z else room2
            bottom_room = room2 if top_room == room1 else room1
            
            min_x = self.agent.pos[0] - door_width/2
            min_x = np.clip(min_x,
                        max(top_room.min_x, bottom_room.min_x) + min_offset,
                        min(top_room.max_x, bottom_room.max_x) - door_width - min_offset)
            
            self.connect_rooms(bottom_room, top_room, min_x=min_x, max_x=min_x + door_width)
        
        current_connection = f"{room1_id}-{room2_id}"
        reverse_connection = f"{room2_id}-{room1_id}"
        
        self.last_created_connection = {
            "from": room1_id,
            "to": room2_id,
            "connection_str": current_connection
        }
        
        special_group = None
        if hasattr(self, 'special_connection_groups'):
            for group_key, connections in self.special_connection_groups.items():
                if current_connection in connections or reverse_connection in connections:
                    special_group = group_key
                    break
        
        if special_group:
            group_connections = self.special_connection_groups[special_group]
            for key, value in self.connections.items():
                if value['connection'] in group_connections:
                    self.connections[key]['created'] = True
        else:
            for key, value in self.connections.items():
                if value['connection'] in [current_connection, reverse_connection]:
                    self.connections[key]['created'] = True
    
    def _get_observation_array(self):
        """Create the observation array with normalized direction vectors"""
        norm_agent_pos = np.array([
            self.agent.pos[0] / self.world_width,
            0,
            self.agent.pos[2] / self.world_depth
        ])
        
        norm_agent_pos[0] = (norm_agent_pos[0] + 1) / 2
        norm_agent_pos[2] = (norm_agent_pos[2] + 1) / 2

        norm_terminal_pos = np.array(self.terminal_location) / np.array([self.world_width, self.world_depth])
        norm_terminal_pos[0] = (norm_terminal_pos[0] + 1) / 2
        norm_terminal_pos[1] = (norm_terminal_pos[1] + 1) / 2

        dx = self.terminal_location[0] - self.agent.pos[0]
        dz = self.terminal_location[1] - self.agent.pos[2]
        
        direction_vector = np.array([dx, -dz])
        direction_length = np.linalg.norm(direction_vector)
        if direction_length > 0:
            direction_vector = direction_vector / direction_length
        else:
            direction_vector = np.array([1.0, 0.0])
        
        agent_dir_vec = np.array([
            np.cos(self.agent.dir),
            np.sin(self.agent.dir)
        ])
        
        dot_product = np.dot(direction_vector, agent_dir_vec)
        angle_difference = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        cross_z = agent_dir_vec[0] * direction_vector[1] - agent_dir_vec[1] * direction_vector[0]
        if cross_z < 0:
            angle_difference = -angle_difference
        
        self.last_angle_difference = angle_difference
        self.last_dot_product = dot_product


        distance_to_terminal = np.linalg.norm(
            np.array([self.agent.pos[0], self.agent.pos[2]]) - 
            np.array([self.terminal_location[0], self.terminal_location[1]])
        )
        
        norm_dist_term = self._normalize_terminal_distance(distance_to_terminal)

        right_dist, left_dist, forward_dist, forward_right_dist, forward_left_dist = self.get_lidar_measurements()

        max_lidar_dist = 10.0
        norm_right = right_dist / max_lidar_dist
        norm_left = left_dist / max_lidar_dist
        norm_forward = forward_dist / max_lidar_dist
        norm_forward_right = forward_right_dist / max_lidar_dist
        norm_forward_left = forward_left_dist / max_lidar_dist

        current_room = self._get_current_room()
        room_category = self._get_room_category(current_room)
        
        return np.array([
            norm_agent_pos[0], norm_agent_pos[2],
            norm_terminal_pos[0], norm_terminal_pos[1],
            norm_dist_term,
            agent_dir_vec[0], agent_dir_vec[1],
            direction_vector[0], direction_vector[1],
            dot_product,
            norm_right, norm_left, norm_forward,
            norm_forward_right, norm_forward_left,
            self.step_count / self.max_episode_steps,
            self.stagnant_steps / 100,
            room_category,
        ], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode"""
        self.has_reached_hallway = False
        self.hallway_reward_given = False
        self.stagnant_steps = 0
        self.non_hallway_steps = 0
        self.reached_terminal_area = False
        
        if seed is not None:
            self.seed(seed)
        else:
            import time
            new_seed = int(time.time() * 1000) % 100000
            self.seed(new_seed)

        observation = super().reset(seed=self.seed_value)
        
        # Place agent (implementation depends on child class)
        self._place_agent()
        
        for connection in self.connections:
            self.connections[connection]['created'] = False

        current_room = self._get_current_room()
        self.started_in_hallway = current_room == 'roomD'
        self.previous_room = current_room

        if self.started_in_hallway:
            self.has_reached_hallway = True
            self.hallway_reward_given = True

        if self.terminal_location is None:
            self.set_terminal_location()

        distance_to_terminal = np.linalg.norm(
            np.array([self.agent.pos[0], self.agent.pos[2]]) - 
            np.array([self.terminal_location[0], self.terminal_location[1]])
        )
        self.previous_distance_to_terminal = distance_to_terminal

        observation_array = self._get_observation_array()
        
        return observation_array, {}
    
    def _place_agent(self):
        """
        Place the agent in the environment.
        Can be overridden by child classes for different placement strategies.
        """
        raise NotImplementedError("Child classes must implement _place_agent()")
    
    def seed(self, seed=None):
        """Seed the environment's random number generator"""
        self.seed_value = seed
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
    
    def step(self, action):
        """Execute an action and return the new state, reward, etc."""
        previous_agent_pos = np.array(self.agent.pos)
        observation, reward, terminated, truncated, info = super().step(action)

        # Initialize reward components
        reward_orientation = 0
        reward_distance_terminal = 0
        punishment_distance_terminal = 0
        punishment_time = 0
        reward_hallway = 0
        reward_connection = 0
        reward_terminal = 0
        punishment_terminal = 0
        punishment_room = 0
        penalty_stagnation = 0
        wall_collision_penalty = 0

        agent_nose_pos = self.agent.pos + self.agent.dir_vec * self.agent.radius
        current_agent_pos = np.array(self.agent.pos)
        
        distance_to_terminal = np.linalg.norm(
            np.array([self.agent.pos[0], self.agent.pos[2]]) - 
            np.array([self.terminal_location[0], self.terminal_location[1]])
        )

        direction_vector = np.array([
            self.terminal_location[0] - self.agent.pos[0],
            -(self.terminal_location[1] - self.agent.pos[2])
        ])
        
        direction_length = np.linalg.norm(direction_vector)
        if direction_length > 0:
            direction_vector = direction_vector / direction_length
        
        agent_dir_vec = np.array([
            np.cos(self.agent.dir),
            np.sin(self.agent.dir)
        ])
        
        dot_product = np.dot(direction_vector, agent_dir_vec)
        angle_difference = np.arccos(np.clip(dot_product, -1.0, 1.0))

        last_angle_difference = self.last_angle_difference
        punishment_time -= 0.5

        position_changed = np.linalg.norm(current_agent_pos - previous_agent_pos) > 0.1

        if position_changed:
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1

        if self.stagnant_steps >= 100:
            penalty_stagnation = -100
            info['penalty_stagnation'] = penalty_stagnation
            info['stagnation_penalty_applied'] = True
            self.stagnant_steps = 0

        if dot_product < np.cos(np.pi / 9):
            reward_orientation -= 0.1

        if position_changed:
            self.previous_distance_to_terminal = distance_to_terminal

        current_room = self._get_current_room()
        is_in_hallway = current_room == 'roomD'
        
        if is_in_hallway:
            reward_hallway += 0.05
            self.non_hallway_steps = 0
            
            first_hallway_visit = not self.has_reached_hallway
            
            if (first_hallway_visit and
                not self.started_in_hallway and
                not self.hallway_reward_given):
                self.hallway_reward_given = True
            
            self.has_reached_hallway = True
        else:
            self.non_hallway_steps += 1
            
            if self.non_hallway_steps >= 100:
                self.non_hallway_steps = 0


        new_connection = self._agent_touches_wall(agent_nose_pos)
        if new_connection is not None and not self.connections[new_connection]['created']:
            self._create_doors(new_connection)
            self._gen_static_data()
            self._render_static()
            info['created_door'] = True
        
            connection_info = self.connections[new_connection]['connection'].split('-')
            info['door_connection_from'] = connection_info[0]
            info['door_connection_to'] = connection_info[1]

        wall_collision_penalty = 0
        
        # Set threshold based on room
        collision_threshold = 0.1 if current_room == 'roomD' else 0.5
        is_touching_connection = new_connection is not None

        if not is_touching_connection:
            right_dist, left_dist, forward_dist, forward_right_dist, forward_left_dist = self.get_lidar_measurements()
            
            if (forward_dist < collision_threshold or 
                right_dist < collision_threshold or 
                left_dist < collision_threshold or
                forward_right_dist < collision_threshold or
                forward_left_dist < collision_threshold):
                
                closest_dist = min(forward_dist, right_dist, left_dist, forward_right_dist, forward_left_dist)
                wall_collision_penalty = -1 * (1 - (closest_dist / collision_threshold))
                
                info['collision'] = True
                info['collision_penalty'] = wall_collision_penalty

        if not self.has_reached_hallway and self.step_count >= self.steps_until_hallway:
            truncated = True
            info['truncation_reason'] = 'hallway_timeout'
            punishment_room = -100
            info['punishment_room'] = punishment_room

        #terminal_x, terminal_z = self.terminal_location
        #if (terminal_x - 0.2 <= agent_nose_pos[0] <= terminal_x + 0.2 and 
        #    terminal_z - 0.2 <= agent_nose_pos[2] <= terminal_z + 0.2):
        #    terminated = True
        #    self.reached_terminal_area = True
        #    reward_terminal = 500
        #    info['reached_terminal'] = True

        # Check if agent is in terminal zone (orientation-aware)
        if self._is_agent_in_terminal_zone(agent_nose_pos):
            terminated = True
            self.reached_terminal_area = True
            reward_terminal = 100
            info['reached_terminal'] = True

        if self.step_count >= self.max_episode_steps:
            truncated = True
            punishment_terminal -= 100
            info['max_steps_reached'] = True

        raw_reward_components = {
            'reward_orientation': reward_orientation,
            'reward_distance': reward_distance_terminal,
            'punishment_distance': punishment_distance_terminal,
            'penalty_stagnation': penalty_stagnation,
            'punishment_time': punishment_time,
            'reward_hallway': reward_hallway,
            'reward_connection': reward_connection,
            'reward_terminal': reward_terminal,
            'punishment_terminal': punishment_terminal,
            'punishment_room': punishment_room,
            'wall_collision_penalty': wall_collision_penalty
        }
        
        reward = (
            self.reward_scales['reward_orientation_scale'] * reward_orientation +
            self.reward_scales['reward_distance_scale'] * reward_distance_terminal +
            self.reward_scales['punishment_distance_scale'] * punishment_distance_terminal +
            self.reward_scales['penalty_stagnation_scale'] * penalty_stagnation +
            self.reward_scales['punishment_time_scale'] * punishment_time +
            self.reward_scales['reward_hallway_scale'] * reward_hallway +
            self.reward_scales['reward_connection_scale'] * reward_connection +
            self.reward_scales['reward_terminal_scale'] * reward_terminal +
            self.reward_scales['punishment_terminal_scale'] * punishment_terminal +
            self.reward_scales['punishment_room_scale'] * punishment_room +
            self.reward_scales['wall_collision_scale'] * wall_collision_penalty 
        )

        info.update(raw_reward_components)
        info.update({
            'episode_rewards': reward,
            'step_count': self.step_count,
            'current_room': current_room,
            'reached_hallway': self.has_reached_hallway 
        })

        observation_array = self._get_observation_array()
        
        return observation_array, reward, terminated, truncated, info
