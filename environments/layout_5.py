"""
Layout 5: Fifth environment layout
- no obstacles yet (Jan 26)
- for testing atm
"""

import numpy as np
import random
from miniworld.entity import Box
from environments.base_env import BaseEscapeEnv


class Layout5aEnv(BaseEscapeEnv):
    """Complex layout with bigger rooms and siding hallway."""
    
    def _gen_world(self):
        """Generate the world layout"""
        self.agent.radius = 0.25

        # Create the hallway that wraps around - L-shaped
        self.roomD = self.add_rect_room(min_x=12, max_x=14, min_z=0, max_z=20.2)
        self.roomD2 = self.add_rect_room(min_x=0, max_x=12, min_z=0, max_z=2)

        # Create center room
        self.roomA = self.add_rect_room(min_x=0, max_x=5.9, min_z=2.2, max_z=6)
        self.roomA2 = self.add_rect_room(min_x=6.1, max_x=11.8, min_z=2.2, max_z=6)

        # Create top room
        self.roomB = self.add_rect_room(min_x=6.8, max_x=11.8, min_z=6.2, max_z=10)
        self.roomB2 = self.add_rect_room(min_x=6.8, max_x=11.8, min_z=10.2, max_z=14)

        # Create right room
        self.roomC = self.add_rect_room(min_x=3, max_x=11.8, min_z=14.2, max_z=17.1)
        self.roomC2 = self.add_rect_room(min_x=3, max_x=11.8, min_z=17.3, max_z=20.2)

        self.rooms.extend([self.roomA, self.roomA2, self.roomB, self.roomB2, self.roomC, self.roomC2, self.roomD, self.roomD2])

        # Place the agent at the starting position
        self.place_entity(
            self.agent,
            pos=[1.0, 0, 1.0],
            dir=0
        )

        # Define room pairs for possible connections
        self.room_pairs = [
            (self.roomD, self.roomD2, 'D', 'D2'),  # Permanent connection
            (self.roomA, self.roomA2, 'A', 'A2'),
            (self.roomA, self.roomD2, 'A', 'D2'),
            (self.roomA2, self.roomD2, 'A2', 'D2'),
            (self.roomA2, self.roomD, 'A2', 'D'),
            (self.roomA2, self.roomB, 'A2', 'B'),
            (self.roomB, self.roomB2, 'B', 'B2'),
            (self.roomB2, self.roomD, 'B2', 'D'),
            (self.roomB, self.roomD, 'B', 'D'),
            (self.roomB2, self.roomC, 'B2', 'C'),
            (self.roomC, self.roomC2, 'C', 'C2'),
            (self.roomC2, self.roomD, 'C2', 'D'),
            (self.roomC, self.roomD, 'C', 'D'),
        ]

        # Add boxes as obstacles
        #self._add_boxes()

        # Generate connection points - D-D2 is marked as permanent (always created)
        self.connections = self.generate_connections(self.room_pairs, [('D', 'D2')])
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
        num_boxes_x = 2
        num_boxes_z = 4
        
        small_box_size = [
            roomA_cluster_width / num_boxes_x,
            roomA_cluster_height,
            roomA_cluster_depth / num_boxes_z
        ]
        
        boxA_base_positions = [
            [2, 0, 1],
            [5, 0, 1],
            [8, 0, 1],
            [2, 0, 5.0],
            [5, 0, 5.0],
            [8, 0, 5.0],
        ]
        
        # Generate smaller boxes for Room A
        for base_pos in boxA_base_positions:
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
                    box.pos = pos
                    box.dir = 0
                    self.boxes.append(box)
                    self.place_entity(box, pos=pos, dir=0, room=self.roomA)

        # Room C boxes
        boxC_base_positions = [
            [2, 0, 15.2],
            [5, 0, 15.2],
            [8, 0, 15.2],
            [2, 0, 19.2],
            [5, 0, 19.2],
            [8, 0, 19.2],
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
                    box.pos = pos
                    box.dir = 0
                    self.boxes.append(box)
                    self.place_entity(box, pos=pos, dir=0, room=self.roomC)
        
        # For room B
        num_boxes_x_B = 4
        num_boxes_z_B = 2
        
        small_box_size_B = [
            roomBC_cluster_width / num_boxes_x_B,
            roomBC_cluster_height,
            roomBC_cluster_depth / num_boxes_z_B
        ]
        
        # Room B boxes
        boxB_base_positions = [
            [8.2, 0, 8.5],
            [8.2, 0, 11.5]
        ]
        
        for base_pos in boxB_base_positions:
            for i in range(num_boxes_x_B):
                for j in range(num_boxes_z_B):
                    x_offset = (i - (num_boxes_x_B-1)/2) * small_box_size_B[0]
                    z_offset = (j - (num_boxes_z_B-1)/2) * small_box_size_B[2]
                    
                    pos = [
                        base_pos[0] + x_offset,
                        base_pos[1],
                        base_pos[2] + z_offset
                    ]
                    
                    box = Box(color='grey', size=small_box_size_B)
                    self.boxes.append(box)
                    self.place_entity(box, pos=pos, dir=0, room=self.roomB)
        
        
    
    def _get_room_category(self, room_name):
        """
        Override to handle 4 rooms.
        0: Hallway (roomD)
        1-3: Other rooms (A, B, C)
        4: Unknown
        """
        if room_name == 'roomD' or room_name == 'roomD2':
            return 0
        elif room_name == 'roomA':
            return 1
        elif room_name == 'roomA2':
            return 2
        elif room_name == 'roomB':
            return 3
        elif room_name == 'roomB2':
            return 4
        elif room_name == 'roomC':
            return 5
        elif room_name == 'roomC2':
            return 6
        else:
            return 7
    
    def _get_hallway_terminal_positions(self):
        """
        Get valid terminal positions along the hallway (roomD).
        Returns list of (x, z) tuples.
        """
        positions = []
        
        # Option 1: Single position at bottom center and very top left
        positions.append((0.0, 1.0))
        positions.append((13.0, 20.2))
        #positions.append((13.0, 0.0))
        
        # Option 2: Multiple positions along right edge
        #x_pos = 13
        #z_start = 1
        #z_end = 19.0
        #z_step = 0.5
        
        #z = z_start
        #while z <= z_end:
        #    positions.append((x_pos, z))
        #    z += z_step
        
        #Option 3: Multiple positions along top edge
        #z_pos = 0
        #x_start = 1.0
        #x_end = 13.0
        #x_step = 0.5

        #x = x_start
        #while x <= x_end:
        #    positions.append((x, z_pos))
        #    x += x_step

        return positions
    
    def _place_agent(self):
        """Place the agent using predefined or random positions"""
        if not hasattr(self, 'all_positions'):
            self.all_positions = [
                # Room A (top)
                ([11.5, 0, 1], 0),
                ([11.5, 0, 1], 0),
                ([12, 0, 1], 0),
                ([12, 0, 1], 0),
                ([12, 0, 1], 0),
                ([12, 0, 1], 0),
                # Room B (middle)
                ([8, 0, 7.25], -np.pi/2),
                ([8, 0, 9.5], -np.pi/2),
                ([8, 0, 12.75], np.pi/2),
                # Room C (bottom)
                ([0.5, 0, 15.2], 0),
                ([3.5, 0, 15.2], 0),
                ([6.5, 0, 15.2], 0),
                ([0.5, 0, 19.2], 0),
                ([3.5, 0, 19.2], 0),
                ([6.5, 0, 19.2], 0),
            ]
            
            self.room_position_ranges = {
                0: (0, 6),     # Room D: positions 0-3
                1: (6, 9),     # Room A: positions 4-7
                2: (9, 15),    # Room A2: positions 8-11
                3: (15, 18),   # Room B: positions 12-14
                4: (18, 21),   # Room B2: positions 15
                5: (21, 27),   # Room C: positions 18-20
            }
            
            self.room_configs = [
                {'min_x': 0.4, 'max_x': 5.5, 'min_z': 2.6, 'max_z': 5.6},  # Room A
                {'min_x': 6.5, 'max_x': 11.4, 'min_z': 2.6, 'max_z': 5.6},  # Room A2
                {'min_x': 7.2, 'max_x': 11.4, 'min_z': 6.6, 'max_z': 9.6},    # Room B
                {'min_x': 7.2, 'max_x': 11.4, 'min_z': 10.6, 'max_z': 13.6},  # Room B2
                {'min_x': 3.4, 'max_x': 11.4, 'min_z': 14.6, 'max_z': 16.7},  # Room C
                {'min_x': 3.4, 'max_x': 11.4, 'min_z': 17.7, 'max_z': 19.8},  # Room C2
            ]
        
        self.room_episode_counts = [3, 3, 3, 3, 3, 3]  # Number of episodes per room before switching

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
                room = self.roomA2
            elif room_index == 2:
                room = self.roomB
            elif room_index == 3:
                room = self.roomB2
            elif room_index == 4:
                room = self.roomC
            else:  # room_index == 5
                room = self.roomC2
            
            self.place_entity(
                self.agent,
                room=room,
                dir=random.uniform(-np.pi, np.pi)
            )
        #else:
        #    config = self.room_configs[room_index]
        #    
        #    # Random position within configured boundaries
        #    x = random.uniform(config['min_x'], config['max_x'])
        #    z = random.uniform(config['min_z'], config['max_z'])
        #    
        #    self.place_entity(
        #        self.agent,
        #        pos=[x, 0, z],
        #        dir=random.uniform(-np.pi, np.pi)
        #    )


class RobustNoRenderLayout5a(Layout5aEnv):
    """
    No-render version of Layout5 for headless training.
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

