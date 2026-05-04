"""
Layout 2: Alternative environment layout
- 4 rooms: A (top-left), B (top-right), C (bottom-right), D (hallway on bottom)
- Different room arrangements and box configurations
"""

import numpy as np
import random
from miniworld.entity import Box
from environments.base_env import BaseEscapeEnv


class Layout2aEnv(BaseEscapeEnv):
    """Alternative layout with hallway on the bottom."""
    
    def __init__(self, max_episode_steps, steps_until_hallway, reward_scales=None, **kwargs):
        """Initialize with correct world dimensions for Layout 2."""
        # Layout 2 rooms extend to 14.0 x 14.4
        # Set dimensions BEFORE calling super().__init__()
        self.world_width = 14.0
        self.world_depth = 14.4
        super().__init__(max_episode_steps, steps_until_hallway, reward_scales, **kwargs)
    
    def _gen_world(self):
        """Generate the world layout"""
        self.agent.radius = 0.25

        # Create the hallway along the bottom
        self.roomD = self.add_rect_room(min_x=0, max_x=14, min_z=6.2, max_z=8.2)

        # Create the top-left room
        self.roomA = self.add_rect_room(min_x=0, max_x=14, min_z=0, max_z=6)

        # Create top-right room
        self.roomB = self.add_rect_room(min_x=0, max_x=6.9, min_z=8.4, max_z=14.4)

        # Create bottom-right room
        self.roomC = self.add_rect_room(min_x=7.1, max_x=14, min_z=8.4, max_z=14.4)

        self.rooms.extend([self.roomA, self.roomB, self.roomC, self.roomD])

        # Place the agent at the starting position
        self.place_entity(
            self.agent,
            pos=[7.0, 0, 1.0],
            dir=0
        )

        # Define room pairs for possible connections
        self.room_pairs = [
            (self.roomA, self.roomD, 'A', 'D'),
            (self.roomB, self.roomD, 'B', 'D'),
            (self.roomB, self.roomC, 'B', 'C'),
            (self.roomC, self.roomD, 'C', 'D'),
        ]

        # Add boxes as obstacles
        self._add_boxes()

        # Generate connection points
        self.connections = self.generate_connections(self.room_pairs, [])
        self._gen_static_data()
    
    def _add_boxes(self):
        """Add boxes as obstacles in each room"""
        self.boxes = []
        
        # Room A cluster parameters
        roomA_cluster_width = 0.6
        roomA_cluster_height = 0.7
        roomA_cluster_depth = 1.4
        
        # Room B and C cluster parameters
        roomBC_cluster_width = 2.5
        roomBC_cluster_height = 0.7
        roomBC_cluster_depth = 0.6
        
        # Room A subdivisions
        num_boxes_x_A = 2
        num_boxes_z_A = 4
        
        small_box_size_A = [
            roomA_cluster_width / num_boxes_x_A,
            roomA_cluster_height,
            roomA_cluster_depth / num_boxes_z_A
        ]
        
        boxA_base_positions = [
            [2.5, 0, 1],
            [5.5, 0, 1],
            [8.5, 0, 1],
            [11.5, 0, 1],
            [2.5, 0, 4],
            [5.5, 0, 4],
            [8.5, 0, 4],
            [11.5, 0, 4]
        ]
        
        # Generate smaller boxes for Room A
        for base_pos in boxA_base_positions:
            for i in range(num_boxes_x_A):
                for j in range(num_boxes_z_A):
                    x_offset = (i - (num_boxes_x_A-1)/2) * small_box_size_A[0]
                    z_offset = (j - (num_boxes_z_A-1)/2) * small_box_size_A[2]
                    
                    pos = [
                        base_pos[0] + x_offset,
                        base_pos[1],
                        base_pos[2] + z_offset
                    ]
                    
                    box = Box(color='grey', size=small_box_size_A)
                    box.pos = pos
                    box.dir = 0
                    self.boxes.append(box)
                    self.place_entity(box, pos=pos, dir=0, room=self.roomA)
        
        # For rooms B and C
        num_boxes_x = 4
        num_boxes_z = 2
        
        small_box_size = [
            roomBC_cluster_width / num_boxes_x,
            roomBC_cluster_height,
            roomBC_cluster_depth / num_boxes_z
        ]
        
        # Room B boxes
        boxB_base_positions = [
            [1.5, 0, 10.5],
            [1.5, 0, 12.75]
        ]
        
        for base_pos in boxB_base_positions:
            for i in range(num_boxes_x):
                for j in range(num_boxes_z):
                    x_offset = (i - (num_boxes_x-1)/2) * small_box_size[0]
                    z_offset = (j - (num_boxes_z-1)/2) * small_box_size[2]
                    
                    pos = [
                        base_pos[0] + x_offset,
                        base_pos[1],
                        base_pos[2] + z_offset
                    ]
                    
                    box = Box(color='grey', size=small_box_size)
                    self.boxes.append(box)
                    self.place_entity(box, pos=pos, dir=0, room=self.roomB)
        
        # Room C boxes
        boxC_base_positions = [
            [12.5, 0, 10.5],
            [12.5, 0, 12.75]  #check if too close to boundary box!!
        ]
        
        for base_pos in boxC_base_positions:
            for i in range(num_boxes_x):
                for j in range(num_boxes_z):
                    x_offset = (i - (num_boxes_x-1)/2) * small_box_size[0]
                    z_offset = (j - (num_boxes_z-1)/2) * small_box_size[2]
                    
                    pos = [
                        base_pos[0] + x_offset,
                        base_pos[1],
                        base_pos[2] + z_offset
                    ]
                    
                    box = Box(color='grey', size=small_box_size)
                    self.boxes.append(box)
                    self.place_entity(box, pos=pos, dir=0, room=self.roomC)
    
    def _get_hallway_terminal_positions(self):
        """
        Get valid terminal positions in the hallway (roomD).
        Returns list of (x, z) tuples.
        """
        positions = []
        
        # Option 1: Single position at sides 
        positions.append((0.0, 7.2))
        positions.append((14.0, 7.2))
        
        return positions
    
    def _place_agent(self):
        """Place the agent using predefined or random positions"""
        if not hasattr(self, 'all_positions'):
            self.all_positions = [
                # Room A (top)
                ([1, 0, 1], 0),
                ([4, 0, 1], 0),
                ([7, 0, 1], 0),
                ([10, 0, 1], 0),
                ([13, 0, 1], np.pi),
                # Room B (left)
                ([2.2, 0, 11.25], np.pi/2),
                ([2.2, 0, 13.5], np.pi/2),
                # Room C (right)
                ([12.2, 0, 11.25], np.pi/2),
                ([12.2, 0, 13.5], np.pi/2),
            ]
            
            self.room_position_ranges = {
                0: (0, 5),    # Room A: positions 0-5
                1: (5, 7),    # Room B: positions 5-7
                2: (7, 9)     # Room C: positions 7-9
            }
            
            self.room_configs = [
                {'min_x': 0.4, 'max_x': 13.6, 'min_z': 0.4, 'max_z': 5.6},   # Room A
                {'min_x': 0.4, 'max_x': 6.5, 'min_z': 9.0, 'max_z': 14.0},   # Room B
                {'min_x': 7.5, 'max_x': 13.6, 'min_z': 9.0, 'max_z': 14.0},  # Room C
            ]

        self.room_episode_counts = [5, 3, 3]

        total_episodes = sum(self.room_episode_counts)
        current_cycle = self.placement_incrementer % total_episodes
        
        # Determine which room based on counts
        cumulative = 0
        room_index = 0
        for i, count in enumerate(self.room_episode_counts):
            cumulative += count
            if current_cycle < cumulative:
                room_index = i
                break
        
        #room_index = (self.placement_incrementer // 3) % len(self.room_configs)
        self.placement_incrementer += 1
        
        use_predefined = random.random() < 0.0
        
        if use_predefined:
            start, end = self.room_position_ranges[room_index]
            position_index = random.randrange(start, end)
            pos, direction = self.all_positions[position_index]
            
            self.place_entity(
                self.agent,
                pos=pos,
                dir=direction
            )
        else:
            if room_index == 0:
                room = self.roomA
            elif room_index == 1:
                room = self.roomB
            else:
                room = self.roomC
            
            self.place_entity(
                self.agent,
                room=room,
                dir=random.uniform(-np.pi, np.pi)
            )


class RobustNoRenderLayout2a(Layout2aEnv):
    """
    No-render version of Layout2a for headless training.
    """
    def __init__(self, max_episode_steps, steps_until_hallway, reward_scales=None, **kwargs):
        kwargs.update({
            'window_width': 1,
            'window_height': 1
        })
        super().__init__(max_episode_steps=max_episode_steps, 
                        steps_until_hallway=steps_until_hallway,
                        reward_scales=reward_scales, **kwargs)
        self.render_mode = 'none'
        
    def render_obs(self, vis_fb=None):
        """Disable observation rendering"""
        return np.zeros((3, 84, 84), dtype=np.uint8)
    
    def _render_static(self):
        """Disable static rendering"""
        pass
        
    def _render_dynamic(self):
        """Disable dynamic rendering"""
        pass


class Layout2bEnv(BaseEscapeEnv):
    """Alternative layout with hallway on the bottom."""
    
    def __init__(self, max_episode_steps, steps_until_hallway, reward_scales=None, **kwargs):
        """Initialize with correct world dimensions for Layout 2b."""
        # Layout 2b rooms extend to 14.0 x 14.4
        # Set dimensions BEFORE calling super().__init__()
        self.world_width = 14.0
        self.world_depth = 14.4
        super().__init__(max_episode_steps, steps_until_hallway, reward_scales, **kwargs)
    
    def _gen_world(self):
        """Generate the world layout"""
        self.agent.radius = 0.25

        # Create the hallway along the bottom
        self.roomD = self.add_rect_room(min_x=0, max_x=14, min_z=6.2, max_z=8.2)

        # Create the top-left room
        self.roomA = self.add_rect_room(min_x=0, max_x=14, min_z=8.4, max_z=14.4)

        # Create top-right room
        self.roomB = self.add_rect_room(min_x=0, max_x=6.9, min_z=0, max_z=6)

        # Create bottom-right room
        self.roomC = self.add_rect_room(min_x=7.1, max_x=14, min_z=0, max_z=6)

        self.rooms.extend([self.roomA, self.roomB, self.roomC, self.roomD])

        # Place the agent at the starting position
        self.place_entity(
            self.agent,
            pos=[7.0, 0, 13.4],
            dir=0
        )

        # Define room pairs for possible connections
        self.room_pairs = [
            (self.roomA, self.roomD, 'A', 'D'),
            (self.roomB, self.roomD, 'B', 'D'),
            (self.roomB, self.roomC, 'B', 'C'),
            (self.roomC, self.roomD, 'C', 'D'),
        ]

        # Add boxes as obstacles
        self._add_boxes()

        # Generate connection points
        self.connections = self.generate_connections(self.room_pairs, [])
        self._gen_static_data()
    
    def _add_boxes(self):
        """Add boxes as obstacles in each room"""
        self.boxes = []
        
        # Room A cluster parameters
        roomA_cluster_width = 0.6
        roomA_cluster_height = 0.7
        roomA_cluster_depth = 1.4
        
        # Room B and C cluster parameters
        roomBC_cluster_width = 2.5
        roomBC_cluster_height = 0.7
        roomBC_cluster_depth = 0.6
        
        # Room A subdivisions
        num_boxes_x_A = 2
        num_boxes_z_A = 4
        
        small_box_size_A = [
            roomA_cluster_width / num_boxes_x_A,
            roomA_cluster_height,
            roomA_cluster_depth / num_boxes_z_A
        ]
        
        boxA_base_positions = [
            [2.5, 0, 13.4],
            [5.5, 0, 13.4],
            [8.5, 0, 13.4],
            [11.5, 0, 13.4],
            [2.5, 0, 10.4],
            [5.5, 0, 10.4],
            [8.5, 0, 10.4],
            [11.5, 0, 10.4]
        ]
        
        # Generate smaller boxes for Room A
        for base_pos in boxA_base_positions:
            for i in range(num_boxes_x_A):
                for j in range(num_boxes_z_A):
                    x_offset = (i - (num_boxes_x_A-1)/2) * small_box_size_A[0]
                    z_offset = (j - (num_boxes_z_A-1)/2) * small_box_size_A[2]
                    
                    pos = [
                        base_pos[0] + x_offset,
                        base_pos[1],
                        base_pos[2] + z_offset
                    ]
                    
                    box = Box(color='grey', size=small_box_size_A)
                    box.pos = pos
                    box.dir = 0
                    self.boxes.append(box)
                    self.place_entity(box, pos=pos, dir=0, room=self.roomA)
        
        # For rooms B and C
        num_boxes_x = 4
        num_boxes_z = 2
        
        small_box_size = [
            roomBC_cluster_width / num_boxes_x,
            roomBC_cluster_height,
            roomBC_cluster_depth / num_boxes_z
        ]
        
        # Room B boxes
        boxB_base_positions = [
            [1.5, 0, 1.75],
            [1.5, 0, 4]
        ]
        
        for base_pos in boxB_base_positions:
            for i in range(num_boxes_x):
                for j in range(num_boxes_z):
                    x_offset = (i - (num_boxes_x-1)/2) * small_box_size[0]
                    z_offset = (j - (num_boxes_z-1)/2) * small_box_size[2]
                    
                    pos = [
                        base_pos[0] + x_offset,
                        base_pos[1],
                        base_pos[2] + z_offset
                    ]
                    
                    box = Box(color='grey', size=small_box_size)
                    self.boxes.append(box)
                    self.place_entity(box, pos=pos, dir=0, room=self.roomB)
        
        # Room C boxes
        boxC_base_positions = [
            [12.5, 0, 1.75],
            [12.5, 0, 4]  #check if too close to boundary box!!
        ]
        
        for base_pos in boxC_base_positions:
            for i in range(num_boxes_x):
                for j in range(num_boxes_z):
                    x_offset = (i - (num_boxes_x-1)/2) * small_box_size[0]
                    z_offset = (j - (num_boxes_z-1)/2) * small_box_size[2]
                    
                    pos = [
                        base_pos[0] + x_offset,
                        base_pos[1],
                        base_pos[2] + z_offset
                    ]
                    
                    box = Box(color='grey', size=small_box_size)
                    self.boxes.append(box)
                    self.place_entity(box, pos=pos, dir=0, room=self.roomC)
    
    def _get_hallway_terminal_positions(self):
        """
        Get valid terminal positions in the hallway (roomD).
        Returns list of (x, z) tuples.
        """
        positions = []
        
        # Option 1: Single position at sides 
        positions.append((0.0, 7.2))
        positions.append((14.0, 7.2))
        
        return positions
    
    def _place_agent(self):
        """Place the agent using predefined or random positions"""
        if not hasattr(self, 'all_positions'):
            self.all_positions = [
                # Room A (top)
                ([1, 0, 13.4], -np.pi/2),
                ([4, 0, 13.4], -np.pi/2),
                ([7, 0, 13.4], -np.pi/2),
                ([10, 0, 13.4], -np.pi/2),
                ([13, 0, 13.4], -np.pi/2),
                # Room B (left)
                ([2.2, 0, 0.75], 0),
                ([2.2, 0, 2.75], 0),
                # Room C (right)
                ([12.2, 0, 0.75], 0),
                ([12.5, 0, 2.75], 0),
            ]
            
            self.room_position_ranges = {
                0: (0, 5),    # Room A: positions 0-5
                1: (5, 7),    # Room B: positions 5-7
                2: (7, 9)     # Room C: positions 7-9
            }
            
            self.room_configs = [
                {'min_x': 0.4, 'max_x': 13.6, 'min_z': 8.8, 'max_z': 14},   # Room A
                {'min_x': 0.4, 'max_x': 6.5, 'min_z': 0.4, 'max_z': 5.6},   # Room B
                {'min_x': 7.5, 'max_x': 13.6, 'min_z': 0.4, 'max_z': 5.6},  # Room C
            ]

        
        self.room_episode_counts = [5, 3, 3]

        total_episodes = sum(self.room_episode_counts)
        current_cycle = self.placement_incrementer % total_episodes
        
        # Determine which room based on counts
        cumulative = 0
        room_index = 0
        for i, count in enumerate(self.room_episode_counts):
            cumulative += count
            if current_cycle < cumulative:
                room_index = i
                break
        
        #room_index = (self.placement_incrementer // 3) % len(self.room_configs)
        self.placement_incrementer += 1
        
        use_predefined = random.random() < 0.0
        
        if use_predefined:
            start, end = self.room_position_ranges[room_index]
            position_index = random.randrange(start, end)
            pos, direction = self.all_positions[position_index]
            
            self.place_entity(
                self.agent,
                pos=pos,
                dir=direction
            )
        else:
            if room_index == 0:
                room = self.roomA
            elif room_index == 1:
                room = self.roomB
            else:
                room = self.roomC
            
            self.place_entity(
                self.agent,
                room=room,
                dir=random.uniform(-np.pi, np.pi)
            )


class RobustNoRenderLayout2b(Layout2bEnv):
    """
    No-render version of Layout2a for headless training.
    """
    def __init__(self, max_episode_steps, steps_until_hallway, reward_scales=None, **kwargs):
        kwargs.update({
            'window_width': 1,
            'window_height': 1
        })
        super().__init__(max_episode_steps=max_episode_steps, 
                        steps_until_hallway=steps_until_hallway,
                        reward_scales=reward_scales, **kwargs)
        self.render_mode = 'none'
        
    def render_obs(self, vis_fb=None):
        """Disable observation rendering"""
        return np.zeros((3, 84, 84), dtype=np.uint8)
    
    def _render_static(self):
        """Disable static rendering"""
        pass
        
    def _render_dynamic(self):
        """Disable dynamic rendering"""
        pass