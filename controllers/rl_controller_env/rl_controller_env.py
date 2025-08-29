# Running Webots with command:
# webots --batch --mode=fast --no-rendering /home/quan-vu/Dissertation/my_first_simulation/worlds/obstacles.wbt
import gymnasium as gym
from gymnasium import spaces
from controller import Supervisor
import numpy as np
import time
import math
import os
import cv2
import sys
import matplotlib.pyplot as plt
from controllers.Gobal_planner import a_star_planner
import pandas as pd
import json

CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the log file path to be in the SAME directory as this controller script
DEBUG_LOG_FILE = os.path.join(CONTROLLER_DIR, "env_debug_log.txt")

# A flag to ensure we only clear the log once when the controller class is first loaded
_log_cleared = False

def log_debug_info(message):
    """Appends a message to the debug log file using an absolute path."""
    global _log_cleared
    # Clear the log file only the very first time this function is called in a run
    if not _log_cleared:
        try:
            if os.path.exists(DEBUG_LOG_FILE):
                os.remove(DEBUG_LOG_FILE)
            with open(DEBUG_LOG_FILE, "w") as f:
                f.write(f"Debug log started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            _log_cleared = True
        except Exception as e:
            # Print error to console if logging fails
            print(f"ERROR: Could not clear or create debug log file at {DEBUG_LOG_FILE}. Error: {e}")
            return # Abort if we can't write

    try:
        with open(DEBUG_LOG_FILE, "a") as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"ERROR: Could not write to debug log file. Error: {e}")

def check_and_log_invalid_data(data, name, episode, step):
    """Checks a numpy array or list for NaN/Inf and logs if found."""
    is_invalid = False
    if data is None:
        log_message = (f"CRITICAL [Ep:{episode}, Step:{step}] - Data for '{name}' is None!")
        is_invalid = True
    # Convert to numpy array for checking if it's a list
    if not isinstance(data, np.ndarray):
        data = np.array(data)
        
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        log_message = (f"CRITICAL [Ep:{episode}, Step:{step}] - Invalid data found in '{name}': {data}")
        is_invalid = True

    if is_invalid:
        print(f"\n{log_message}")
        log_debug_info(log_message)
        return True
    return False

class WebotsRobotEnvironment(gym.Env):
    """
    A custom Gymnasium environment for controlling a Webots robot.
    Uses a privileged observer for obstacle data and Lidar for collision detection.
    Connects to a robot with an <extern> controller.
    """
    def __init__(self, render_mode='none',failure_log_path=None, is_testing=False):
        super(WebotsRobotEnvironment, self).__init__()

        # Plot configs\
        self.is_testing = is_testing
        self.render_mode = render_mode
        self.fig = None
        self.ax_map = None

        # Data collection
        self.current_start_cell = None
        self.current_goal_cell = None
        self.current_obstacle_params = []

        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.robot_node = self.supervisor.getSelf()
        if self.robot_node is None:
            sys.exit("ERROR: Could not get robot node. Is the MiR100 controller set to <extern> and is it a Supervisor?")

        # --- Get Handles to World Obstacles ---
        self.obstacle_nodes = []
        obstacle_def_names = ["STA_OBJ_1", "STA_OBJ_2"]
        for name in obstacle_def_names:
            node = self.supervisor.getFromDef(name)
            if node: self.obstacle_nodes.append(node)
            else: print(f"Warning: Could not find static obstacle with DEF name: {name}")

        self.dynamic_obstacle_nodes = []
        dyn_obs_def_names = ["DYN_OBJ_1"]
        for name in dyn_obs_def_names:
            node = self.supervisor.getFromDef(name)
            if node: self.dynamic_obstacle_nodes.append(node)
            else: print(f"Warning: Could not find dynamic obstacle with DEF name: {name}")
        
        self.MAX_STEPS_PER_EPISODE = 5000

        # --- Initialize Devices ---
        self.left_motor = self.supervisor.getDevice("middle_left_wheel_joint")
        self.right_motor = self.supervisor.getDevice("middle_right_wheel_joint")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.gps = self.supervisor.getDevice("gps")
        self.gps.enable(self.timestep)
        self.imu = self.supervisor.getDevice("inertial unit")
        self.imu.enable(self.timestep)
        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.bumper = self.supervisor.getDevice("touch sensor")
        self.bumper.enable(self.timestep)
        
        # --- Define Action and Observation Spaces for Privileged Observer ---
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation space:
        self.NUM_LIDAR_BINS = 32 # Define the number of lidar bins/sectors
        # Self Kinematics (2) + Path Errors (2) + Privileged Obstacles(N*M) + Lidar(K) + Future Path(P*Q)
        num_obs = 2 + 2 + 2 + self.NUM_LIDAR_BINS 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float32)
        print(f"Observation space configured with {num_obs} dimensions.")
        
        # --- Configuration and State Variables ---
        self.max_linear_velocity = 1.0
        self.max_angular_velocity = np.pi
        self.wheel_radius = 0.0625
        self.distance_to_center = 0.2226

        self.WORLD_X_METERS = 30.0
        self.WORLD_Y_METERS = 30.0
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        map_path = os.path.join(script_dir, "..", "Creating_map", "map_outputs", "factory_map_edited.npy")
        self.occupancy_grid = np.load(os.path.normpath(map_path))
        print(f"C-Space map loaded successfully from: {os.path.normpath(map_path)}")
        kernel = np.ones((3,3), np.uint8)
        self.c_space_grid = cv2.dilate(self.occupancy_grid, kernel, iterations=1)
        self.cells_per_meter = (self.c_space_grid.shape[0]-2) / self.WORLD_X_METERS
        self.dist_transform = cv2.distanceTransform(cv2.bitwise_not(self.c_space_grid), cv2.DIST_L2, 5)

        self.global_path_world = []
        self.target_waypoint_index = 0
        self.obstacle_idx = 0
        self.previous_distance_to_waypoint = float('inf')
        self.static_obstacle_phase = False
        self.dynamic_obstacle_phase = False
        self.episode_count = 0 
        self.current_start_yaw = 0.0
        self.previous_position = np.array([0.0, 0.0])

        self.lookahead_distance_nodes = 5   # node
        self.waypoint_tolerance = 0.5       # meter
        self.path_deviation_threshold = 0.5 # meter
        self.stagnation_counter = 0
        self.accumulated_angle = 0.0        # rad
        self.last_omega_sign = 0
        self.was_just_avoiding = False
        self.blocked_path_indices = set()
        self.current_idx = 0
        self.segment_lengths = np.array([])
        self.cumulative_segment_lengths = np.array([])

        # self.visited_cells = set()

        self.failure_replay_index = 0
        self.robot_trajectory = []
        self.all_episode_trajectories = []
        self.plot_update_freq = 10  # Update plot every N steps
        self.failure_log_path = failure_log_path
        self.failure_scenarios = []
        if self.failure_log_path and os.path.exists(self.failure_log_path):
            try:
                # Use pandas to easily read the CSV
                df = pd.read_csv(self.failure_log_path)
                # Convert the DataFrame to a list of dictionaries
                self.failure_scenarios = df.to_dict('records')
                print(f"Successfully loaded {len(self.failure_scenarios)} failure scenarios from '{self.failure_log_path}'.")
            except Exception as e:
                print(f"Warning: Could not load or parse failure log at '{self.failure_log_path}'. Error: {e}")
        else:
            if self.failure_log_path:
                 print(f"Warning: Failure log file not found at '{self.failure_log_path}'. Will only use random scenarios.")

        if self.render_mode == 'plot':
            self._plot_init()
    
    def _plot_init(self):
        """Initializes the Matplotlib figure and axes for plotting."""
        if self.fig is None: # Only create if it doesn't exist
            plt.ion() # Turn on interactive mode
            self.fig = plt.figure(figsize=(6, 6)) # Create one figure
            self.ax_map = self.fig.add_subplot(1, 1, 1) # Add one subplot for the map
            self.fig.suptitle('RL Agent Navigation', fontsize=12)
            print("Matplotlib plotting initialized.")

    def _update_plot(self):
        """Updates and renders the Matplotlib plot with all new features."""
        if self.render_mode != 'plot' or self.fig is None:
            return

        # --- 1. Get current data for plotting ---
        start_pos_world = self.global_path_world[0] if self.global_path_world else None
        end_pos_world = self.global_path_world[-1] if self.global_path_world else None
        robot_pos_3d = self.gps.getValues()
        robot_pos_2d = [robot_pos_3d[0], robot_pos_3d[1]]

        # --- 2. Update the Plot ---
        self.ax_map.clear()
        
        # Set plot limits and aspect ratio (as before)
        margin = 1.0
        self.ax_map.set_xlim(-(self.WORLD_X_METERS+1) / 2 - margin, (self.WORLD_X_METERS+1) / 2 + margin)
        self.ax_map.set_ylim(-(self.WORLD_Y_METERS+1) / 2 - margin, (self.WORLD_Y_METERS+1) / 2 + margin)
        self.ax_map.set_aspect('equal', adjustable='box')

        # --- MODIFICATION: Create and Display Distinct C-Space Map ---
        
        c_space_visual = np.zeros_like(self.occupancy_grid, dtype=np.uint8)
        c_space_visual[self.c_space_grid == 1] = 1 # 1 = Inflated Zone (grey)
        c_space_visual[self.occupancy_grid == 1] = 2 # 2 = True Obstacle (black)
        
        from matplotlib.colors import ListedColormap
        cmap_cspace = ListedColormap(['white', 'grey', 'black'])
        
        map_extent = [-(self.WORLD_X_METERS+1) / 2, (self.WORLD_X_METERS+1) / 2,
                      (self.WORLD_Y_METERS+1) / 2, -(self.WORLD_Y_METERS+1) / 2]
        self.ax_map.imshow(c_space_visual, cmap=cmap_cspace, origin='lower', extent=map_extent, interpolation='nearest')
        # --- END MODIFICATION ---

        # --- MODIFICATION: Add White Grid ---
        grid_x_ticks = np.arange(-(self.WORLD_X_METERS+1) / 2, (self.WORLD_X_METERS+1) / 2 + 1, 0.5)
        grid_y_ticks = np.arange(-(self.WORLD_Y_METERS+1) / 2, (self.WORLD_Y_METERS+1) / 2 + 1, 0.5)
        self.ax_map.set_xticks(grid_x_ticks)
        self.ax_map.set_yticks(grid_y_ticks)
        self.ax_map.grid(which='major', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        self.ax_map.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        self.ax_map.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        # --- END MODIFICATION ---

        # Plot the A* path (as before)
        if self.global_path_world:
            path_x, path_y = zip(*self.global_path_world)
            self.ax_map.plot(path_x, path_y, 'b--', linewidth=1, alpha=1.0, label='A* Path')

        # --- MODIFICATION: Plot ALL previous trajectories ---
        for i, trajectory in enumerate(self.all_episode_trajectories):
            if len(trajectory) > 1:
                traj_x, traj_y = zip(*trajectory)
                # Use a semi-transparent color for old paths
                self.ax_map.plot(traj_x, traj_y, 'g-', linewidth=1.5, alpha=0.2)
        # --- END MODIFICATION ---

        # Plot the CURRENT trajectory
        if self.robot_trajectory and len(self.robot_trajectory) > 1:
            curr_traj_x, curr_traj_y = zip(*self.robot_trajectory)
            self.ax_map.plot(curr_traj_x, curr_traj_y, 'm-', linewidth=1.5, label='Current Trajectory')

        # Plot Start, Goal, and Obstacles with smaller size
        obstacle_color = 'brown'
        obstacle_radius_world = 0.25 # Smaller radius to fit in one cell
        if self.current_obstacle_positions:
            for i, obs_pos_world in enumerate(self.current_obstacle_positions):
                label = 'Obstacle' if i == 0 else ""
                obstacle_circle = plt.Circle(obs_pos_world, obstacle_radius_world, 
                                             color=obstacle_color, alpha=0.9, label=label)
                self.ax_map.add_patch(obstacle_circle)
        
        if start_pos_world:
            self.ax_map.plot(start_pos_world[0], start_pos_world[1], 'go', markersize=7, label='Start')
        if end_pos_world:
            self.ax_map.plot(end_pos_world[0], end_pos_world[1], 'ro', markersize=7, label='End')

        # Plot current robot position on top
        self.ax_map.plot(robot_pos_2d[0], robot_pos_2d[1], 'D', markersize=6, label='Robot')

        # Update legend
        # handles, labels = self.ax_map.get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # self.ax_map.legend(by_label.values(), by_label.keys(), fontsize='x-small')

        handles, labels = self.ax_map.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # This moves the legend outside the plot to the top right corner
        # and increases the font size.
        self.ax_map.legend(
            by_label.values(), 
            by_label.keys(), 
            fontsize='medium', # Changed from 'x-small' to 'small'
            loc='upper left', 
            bbox_to_anchor=(1.02, 1.0) # Moves the legend anchor outside the axes
)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render(self):
        if self.render_mode == 'plot':
            if self.current_step % self.plot_update_freq == 0:
                self._update_plot()

    def close(self):
        """Closes the Matplotlib plot window."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax_map = None
            print("Matplotlib plot closed.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.robot_trajectory:
            self.all_episode_trajectories.append(list(self.robot_trajectory))

        self.robot_trajectory.clear()
        self.current_step = 0
        self.episode_count += 1
        
        # Reset state variables
        self.stagnation_counter = 0
        self.accumulated_angle = 0.0
        self.current_obstacle_positions = []
        self.last_omega_sign = 0
        self.was_just_avoiding = False
        self.obstacle_idx = 0
        # self.visited_cells.clear()
        yaw_holder = 0.0
        self.blocked_path_indices.clear()
        
        self.supervisor.simulationResetPhysics()
        
        # Hide all movable obstacles at the start
        for node in self.obstacle_nodes + self.dynamic_obstacle_nodes:
            node.getField('translation').setSFVec3f([0, 0, -10])

        # --- 1. DETERMINE SCENARIO PARAMETERS (Start, Goal, Obstacles) ---
        start_cell, goal_cell = None, None
        obstacle_params= []
        path_cells = None
        redoing_flag = False

        # Coin flip: 50% chance to replay a failure, if any exist
        if self.failure_scenarios and np.random.rand() < 0.1 and self.episode_count >= 1000:
            print(f"\n--- Starting Episode #{self.episode_count} (Replaying Failure Scenario) ---")
            scenario = self.failure_scenarios[self.failure_replay_index]
            # Increment the index for the next time
            self.failure_replay_index += 1
            
            if self.failure_replay_index >= len(self.failure_scenarios):
                self.failure_replay_index = 0
                print("  - Reached end of failure log. Looping back to the beginning.")
            redoing_flag = True
            start_cell = (scenario['start_cell_row'], scenario['start_cell_col'])
            goal_cell = (scenario['goal_cell_row'], scenario['goal_cell_col'])
            yaw_holder = scenario.get('start_yaw_rad', 0.0)
            obstacle_params = json.loads(scenario['obstacle_params_json'])
            
            print(f"  - Replaying scenario: start={start_cell}, goal={goal_cell}, obstacles = {self.current_obstacle_params}" )
        else:
            print(f"\n--- Starting Episode #{self.episode_count} (New Random Scenario) ---")
            while True:                
                start_cell = a_star_planner.find_random_free_space(self.c_space_grid)
                goal_cell = a_star_planner.find_random_free_space(self.c_space_grid)
                manhattan_dist = abs(start_cell[0] - goal_cell[0]) + abs(start_cell[1] - goal_cell[1])
                if manhattan_dist >= 13:
                    break
            redoing_flag = False  

        # Generate the path using A* planner
        path_cells = a_star_planner.find_path(self.c_space_grid, start_cell, goal_cell, self.dist_transform)
        if not path_cells:
            print("ERROR: No valid path found between start and goal. Resetting environment.")
            return self.reset(seed, options)

        if redoing_flag:
            num_obstacles_to_spawn = len(obstacle_params)
            if num_obstacles_to_spawn > 0:
                print(f'  - Spawning {num_obstacles_to_spawn} obstacles from replayed scenario.')
            for i, params in enumerate(obstacle_params):
                center_node_index, side = params
                if not (0 < center_node_index < len(path_cells) - 1): continue 
        else:
            if self.static_obstacle_phase:
                num_obstacles_to_spawn = 0
                path_length_nodes = len(path_cells)
                min_spawn = 1 if self.is_testing else 0
                if 13 <= path_length_nodes <= 20:
                    num_obstacles_to_spawn = np.random.randint(min_spawn, 2)
                elif path_length_nodes > 20:
                    num_obstacles_to_spawn = np.random.randint(min_spawn, 3)
                
                num_obstacles_to_spawn = min(num_obstacles_to_spawn, len(self.obstacle_nodes))
                    
                if num_obstacles_to_spawn > 0:
                    print(f"INFO: Spawning {num_obstacles_to_spawn} curriculum obstacle(s).")
                    spawn_padding = 6
                    if path_length_nodes > (2 * spawn_padding):
                        valid_spawn_indices = list(range(spawn_padding, path_length_nodes - spawn_padding))
                        
                        for i in range(num_obstacles_to_spawn):
                            if not valid_spawn_indices: break
                            
                            center_node_index = np.random.choice(valid_spawn_indices)
                            indices_to_remove = range(center_node_index - 4, center_node_index + 5)
                            valid_spawn_indices = [idx for idx in valid_spawn_indices if idx not in indices_to_remove]
                            side = np.random.choice([-1, 0, 1])
                            obstacle_params.append((center_node_index, side))

            if self.dynamic_obstacle_phase:
                print(f"INFO: Dynamic Obstacle Phase ACTIVE. Spawning {len(self.dynamic_obstacle_nodes)} dynamic obstacle(s).")
                for node in self.dynamic_obstacle_nodes:
                    # Spawn randomly in a free space cell
                    spawn_pos_cell = a_star_planner.find_random_free_space(self.c_space_grid)
                    spawn_pos_x, spawn_pos_y = self._convert_cell_to_world(spawn_pos_cell)
                    node.getField('translation').setSFVec3f([spawn_pos_x, spawn_pos_y, 0.3])
                    self.current_obstacle_positions.append((spawn_pos_x, spawn_pos_y))
                    # Give it a random velocity
                    velocity = [np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3), 0]
                    node.setVelocity(velocity)
        
                self.current_start_cell = start_cell
        
        self.current_start_cell = start_cell
        self.current_goal_cell = goal_cell
        self.current_obstacle_params = obstacle_params
        self.global_path_world = self._convert_path_to_world(path_cells)
    
        path_points = np.array(self.global_path_world)
        path_vecs = path_points[1:] - path_points[:-1]
        self.segment_lengths = np.linalg.norm(path_vecs, axis=1)
        # Cumulative sum of lengths up to the START of each segment
        self.cumulative_segment_lengths = np.insert(np.cumsum(self.segment_lengths), 0, 0)[:-1]


        num_obstacles_to_place = min(len(obstacle_params),len(self.obstacle_nodes))
        if num_obstacles_to_place > 0:
             print(f"  - Spawning {num_obstacles_to_place} obstacles.")
        for i, params in enumerate(obstacle_params):
            center_node_index, side = params
            self.obstacle_idx = center_node_index
            if not (0 < center_node_index < len(path_cells) - 1): continue # Safety check
                
            # This is your shared spawning equation
            p1 = np.array(self._convert_cell_to_world(path_cells[center_node_index - 1]))
            p2 = np.array(self._convert_cell_to_world(path_cells[center_node_index + 1]))
            path_dir_vec = p2 - p1
            perp_vec = np.array([-path_dir_vec[1], path_dir_vec[0]])
            perp_vec_norm = perp_vec / (np.linalg.norm(perp_vec) + 1e-6)
            offset = 0.5
            center_node_world = np.array(self._convert_cell_to_world(path_cells[center_node_index]))
            spawn_pos = center_node_world + side * offset * perp_vec_norm
            
            # Place the obstacle and record its blocked index
            self.current_obstacle_positions.append(tuple(spawn_pos))
            self.obstacle_nodes[i].getField('translation').setSFVec3f([spawn_pos[0], spawn_pos[1], 0.3])
            self.blocked_path_indices.add(center_node_index)

        # --- 4. SET ROBOT POSE & FINALIZE ---
        self.target_waypoint_index = 0
        robot_start_pos = self.global_path_world[0]
        if redoing_flag:
            start_yaw = yaw_holder  
        else:
            start_yaw = np.random.uniform(-np.pi, np.pi)
        self.current_start_yaw = start_yaw
        self._set_robot_pose(robot_start_pos[0], robot_start_pos[1], self.current_start_yaw)
        # self.visited_cells.add(self._world_to_grid(robot_start_pos))

        if self.supervisor.step(self.timestep) == -1:
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        
        if self.global_path_world and len(self.global_path_world) > 1:
            print("INFO: Performing initial alignment...")
            # Align to the second waypoint in the path (index 1)
            target_waypoint = self.global_path_world[1]
            print(f"INFO: Initial node position: {self.global_path_world[0]}")
            print(f"INFO: Target waypoint for alignment: {target_waypoint}")
            print(f"INFO: Final node position: {self.global_path_world[len(self.global_path_world) - 1]}")
            
            for _ in range(200):
                current_pos = self.gps.getValues() # This will now be the correct start pos
                current_yaw = self.imu.getRollPitchYaw()[2]
                
                dx = target_waypoint[0] - current_pos[0]
                dy = target_waypoint[1] - current_pos[1]

                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    break
                
                angle_to_target = math.atan2(dy, dx)
                angle_error = angle_to_target - current_yaw
                
                if angle_error > math.pi: angle_error -= 2 * math.pi
                if angle_error < -math.pi: angle_error += 2 * math.pi
                
                if abs(angle_error) < 0.1: 
                    break
                
                turn_speed = np.clip(angle_error * 1.5, -self.max_angular_velocity, self.max_angular_velocity)
                self._set_robot_velocities(0.0, turn_speed)
                
                if self.supervisor.step(self.timestep) == -1:
                    return np.zeros(self.observation_space.shape, dtype=np.float32), {}
            
            self._set_robot_velocities(0.0, 0.0)
            print("INFO: Alignment complete.")
            
        # --- STEP 5: RETURN INITIAL OBSERVATION ---
        initial_observation = self._get_observation()
        
        # Initialize previous_distance for the first step's reward calculation
        if self.global_path_world:
            lookahead_point = self._find_lookahead_point(np.array(robot_start_pos))
            self.previous_distance_to_waypoint = np.linalg.norm(np.array(robot_start_pos) - lookahead_point)
        else:
            self.previous_distance_to_waypoint = 0.0

        self.render()
        return initial_observation, {}
    
    def step(self, action):
        linear_vel = ((action[0] + 1) / 2) * self.max_linear_velocity
        angular_vel = action[1] * self.max_angular_velocity
        self._set_robot_velocities(linear_vel, angular_vel)
        new_pos_3d = self.gps.getValues()
        self.robot_trajectory.append([new_pos_3d[0], new_pos_3d[1]])

        if self.supervisor.step(self.timestep) == -1:
            return np.zeros(self.observation_space.shape), 0, True, True, {}
        
        self.current_step += 1
        observation = self._get_observation()
        # --- MODIFICATION: Capture the info dictionary ---
        reward, terminated, truncated, info = self._get_reward(observation)
        # --- END MODIFICATION ---
        self.render()

        return observation, reward, terminated, truncated, info

    def _get_path_projection_info(self, robot_pos_np):
        if not self.global_path_world:
            return robot_pos_np, 0.0, -1

        # Convert the entire path to a NumPy array
        path_points = np.array(self.global_path_world)
        if len(path_points) < 2:
            # If path has 0 or 1 point, projection is the point itself
            dist_to_point = np.linalg.norm(robot_pos_np - path_points[0]) if len(path_points) > 0 else 0.0
            return path_points[0] if len(path_points) > 0 else robot_pos_np, 0.0, 0
        
        # Create vectors for all segments at once
        # p1s are points [0] to [n-2], p2s are points [1] to [n-1]
        p1s = path_points[:-1]
        p2s = path_points[1:]
        path_vecs = p2s - p1s
        
        # Vector from the start of each segment to the robot
        robot_vecs = robot_pos_np - p1s
        
        # Calculate squared length of each path segment
        path_len_sq = np.sum(path_vecs**2, axis=1)

        # Calculate projection parameter 't' for all segments
        # Handle zero-length segments to avoid division by zero
        # np.divide is used for safe division, with a 'where' clause
        t = np.divide(np.einsum('ij,ij->i', robot_vecs, path_vecs), path_len_sq, 
                      out=np.zeros_like(path_len_sq), where=path_len_sq!=0)
        
        # Clamp t to the [0, 1] range to stay on the segments
        t_clamped = np.clip(t, 0, 1)
        
        # Calculate all projection points
        projections = p1s + t_clamped[:, np.newaxis] * path_vecs
        
        # Find the distance from the robot to each projection point
        dist_sq_to_projections = np.sum((robot_pos_np - projections)**2, axis=1)
        
        # Find the index of the segment with the minimum distance
        closest_segment_index = np.argmin(dist_sq_to_projections)
        
        # Get the specific projection point for the closest segment
        projection = projections[closest_segment_index]
        
        # --- Calculate total distance along the path to the projected point ---
        
        # Pre-calculate segment lengths if not already done (can be cached)
        segment_lengths = np.linalg.norm(path_vecs, axis=1)
        
        # Sum of lengths of all segments before the closest one
        dist_before_segment = np.sum(segment_lengths[:closest_segment_index])
        
        # Distance from the start of the closest segment to the projection point
        start_of_closest_segment = p1s[closest_segment_index]
        dist_on_segment = np.linalg.norm(projection - start_of_closest_segment)
        
        distance_along_path = self.cumulative_segment_lengths[closest_segment_index] + dist_on_segment

        return projection, distance_along_path, closest_segment_index

    def _find_lookahead_point(self, robot_pos_np):
        """
        Finds the look-ahead point on the global path for the RL agent to target.
        """
        if not self.global_path_world:
            return robot_pos_np # No path, target is current position

        # Get the robot's projection onto the path to find its current progress
        _projection, _dist_along, closest_segment_index = self._get_path_projection_info(robot_pos_np)
        
        if closest_segment_index == -1:
            return robot_pos_np # Should not happen if path exists

        # Your logic: Look ahead 5 nodes from the *start* of the current segment
        lookahead_index = closest_segment_index + self.lookahead_distance_nodes
        
        # Your logic: Handle the end-of-path case
        # If the lookahead index goes past the end of the path, clamp it to the last waypoint
        if lookahead_index >= len(self.global_path_world):
            lookahead_index = len(self.global_path_world) - 1
            
        lookahead_point = np.array(self.global_path_world[lookahead_index])
        
        return lookahead_point

    def _get_observation(self):
        # --- 1. Get Robot's Own State ---
        robot_pos_3d = self.gps.getValues()
        robot_pos_2d_np = np.array([robot_pos_3d[0], robot_pos_3d[1]])
        robot_vel_vec = self.robot_node.getVelocity()
        robot_yaw = self.imu.getRollPitchYaw()[2]
        current_v = math.sqrt(robot_vel_vec[0]**2 + robot_vel_vec[1]**2)
        current_omega = robot_vel_vec[5]

        # --- 2. Find the Look-Ahead Point to use as the RL Goal ---
        lookahead_point = self._find_lookahead_point(robot_pos_2d_np)

        # --- 3. Calculate Goal Info (relative to the look-ahead point) ---
        dx_goal = lookahead_point[0] - robot_pos_2d_np[0]
        dy_goal = lookahead_point[1] - robot_pos_2d_np[1]
        goal_distance = math.hypot(dx_goal, dy_goal)
        goal_angle = math.atan2(dy_goal, dx_goal) - robot_yaw
        if goal_angle > math.pi: goal_angle -= 2 * math.pi
        if goal_angle < -math.pi: goal_angle += 2 * math.pi

        final_goal_pos = np.array(self.global_path_world[-1])
        dx_final = final_goal_pos[0] - robot_pos_2d_np[0]
        dy_final = final_goal_pos[1] - robot_pos_2d_np[1]
        dist_to_final_goal = math.hypot(dx_final, dy_final)
        angle_to_final_goal = math.atan2(dy_final, dx_final) - robot_yaw
        if angle_to_final_goal > math.pi: angle_to_final_goal -= 2 * math.pi
        if angle_to_final_goal < -math.pi: angle_to_final_goal += 2 * math.pi

        # --- 3. Get Lidar Info (NEW NON-UNIFORM BINNING) ---
        raw_lidar_ranges = self.lidar.getLayerRangeImage(0)
        max_range = self.lidar.getMaxRange()
        lidar_binned = np.full(self.NUM_LIDAR_BINS, max_range, dtype=np.float32)
        
        if raw_lidar_ranges:
            num_raw_readings = len(raw_lidar_ranges)
            front_scan_start_index = num_raw_readings // 4
            front_scan_end_index = front_scan_start_index * 3
            front_scan_raw = raw_lidar_ranges[front_scan_start_index:front_scan_end_index]
            
            num_front_readings = len(front_scan_raw)
            if num_front_readings > 0:
                total_fov_deg = 180.0
                degrees_per_reading = total_fov_deg / num_front_readings

                for i, dist in enumerate(front_scan_raw):
                    if math.isinf(dist) or math.isnan(dist): dist = max_range
                    
                    angle_deg = (i * degrees_per_reading) - (total_fov_deg / 2.0) # Angle from -90 (right) to +90 (left)
                    
                    bin_index = -1
                    # Right peripheral bin (30 degrees)
                    if -90 <= angle_deg < -60:
                        bin_index = 0
                    # Central high-res bins (120 degrees / 30 bins)
                    elif -60 <= angle_deg < 60:
                        bin_index = 1 + int((angle_deg + 60) / 4.0) # Bins 1 to 30
                    # Left peripheral bin (30 degrees)
                    elif 60 <= angle_deg <= 90:
                        bin_index = 31 # The last bin
                    
                    if bin_index != -1:
                        bin_index = np.clip(bin_index, 0, self.NUM_LIDAR_BINS - 1)
                        if dist < lidar_binned[bin_index]:
                            lidar_binned[bin_index] = dist
        
        normalized_lidar = lidar_binned / max_range
        check_and_log_invalid_data(normalized_lidar, "normalized_lidar", self.episode_count, self.current_step)

        # --- 4. Assemble the final NORMALIZED observation vector ---
        dist_normalization_factor = 5.0
        
        observation = np.concatenate([
            [np.clip(goal_distance / dist_normalization_factor, 0, 1.5), goal_angle / np.pi],
            [np.clip(dist_to_final_goal / dist_normalization_factor, 0, 1.5), angle_to_final_goal / np.pi],
            [current_v / self.max_linear_velocity, current_omega / self.max_angular_velocity], # Self Kinematics
            normalized_lidar,        # Already normalized
        ]).astype(np.float32)

        if check_and_log_invalid_data(observation, "FINAL_OBSERVATION", self.episode_count, self.current_step):
            # If the final vector is invalid for any reason, return a safe zero vector
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        if observation.shape[0] != self.observation_space.shape[0]:
            print(f"\nCRITICAL ERROR: Observation shape mismatch! Expected {self.observation_space.shape[0]}, got {observation.shape[0]}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        return observation

    def _get_reward(self, observation):
        terminated = False
        truncated = False
        info = {}
        reward_components = {}

        # Get current robot state once at the start
        robot_pos_np = np.array(self.gps.getValues()[:2])
        # De-normalize velocities from the observation to use for penalties
        current_v = observation[4] * self.max_linear_velocity
        current_omega = observation[5] * self.max_angular_velocity

        # --- 1. TERMINAL REWARD CALCULATION ---
        # A. Check for Final Goal Reached (Robust Distance Check)
        if self.global_path_world:
            final_goal_pos = np.array(self.global_path_world[-1])
            if np.linalg.norm(robot_pos_np - final_goal_pos) < self.waypoint_tolerance:
                reward = 500.0
                terminated = True
                print(f"INFO: Episode {self.episode_count}: Final goal reached!")
                info['termination_reason'] = 'goal_reached'
                return reward, terminated, truncated, info 

        # B. Check for Terminal Failure Condition (Collision)
        if self.bumper.getValue() > 0:
            reward = -300.0 # Scale penalty by progress
            terminated = True
            print(f"INFO: Episode {self.episode_count}: Bumper Collision Detected!")
            info['termination_reason'] = 'collision'
            info['termination_pos'] = (robot_pos_np[0], robot_pos_np[1])
            return reward, terminated, truncated, info

        # C. Timeout Check
        if self.current_step >= self.MAX_STEPS_PER_EPISODE:
            truncated = True
            if not terminated: # Don't add timeout penalty if episode also succeeded
                reward = -50.0
            print(f"INFO: Episode {self.episode_count}: Truncated due to step limit!")
            info['termination_reason'] = 'timeout'
            return reward, terminated, truncated, info

        # --- 2. ONGOING REWARD CALCULATION ---
        # A. Your Path Following Reward
        _proj_curr, dist_along_path_curr, seg_idx_curr = self._get_path_projection_info(robot_pos_np)
        self.current_idx = seg_idx_curr # Update current index for later use
        _proj_prev, dist_along_path_prev, _seg_idx_prev = self._get_path_projection_info(self.previous_position)
        lookahead_point = self._find_lookahead_point(robot_pos_np)
        current_distance_to_lookahead = np.linalg.norm(robot_pos_np - lookahead_point)

        progress = dist_along_path_curr - dist_along_path_prev
        cross_track_error = np.linalg.norm(robot_pos_np - _proj_curr)
        
        # Give a bonus for forward progress, penalize moving backward on the path
        w_progress = 75 # Tunable weight 10
        path_reward = w_progress * progress
        reward_components['path_reward'] = path_reward

        # Check if the robot's actual progress (closest_segment_index) has overtaken its milestone target
        if self.global_path_world and seg_idx_curr >= self.target_waypoint_index:
            waypoints_advanced = (seg_idx_curr + 1) - self.target_waypoint_index
            target_waypoint = np.array(self.global_path_world[self.target_waypoint_index])
            distance_to_target = np.linalg.norm(robot_pos_np - target_waypoint)
            if waypoints_advanced > 0:
                if distance_to_target < self.waypoint_tolerance:
                    print(f"INFO: Episode {self.episode_count}: Progressed past waypoint {self.target_waypoint_index}!")

                self.target_waypoint_index = seg_idx_curr+ 1

        path_rejoin_reward = 0.0
        last_obstacle_idx = -1
        if self.blocked_path_indices:
            last_obstacle_idx = max(self.blocked_path_indices)
        # Check if the flag is set (we just finished avoiding) AND if we are back on track
        if self.current_idx > last_obstacle_idx and not self.was_just_avoiding and last_obstacle_idx != -1:
            if cross_track_error < 0.4: # Your desired threshold
                path_rejoin_reward = 50.0 # A significant "Good Job!" reward
                self.was_just_avoiding = True
                print (f"INFO: self.current_idx: {self.current_idx}, self.obstacle_idx: {self.obstacle_idx}")
                print(f"INFO: Episode {self.episode_count}: Successful path rejoin! (+{path_rejoin_reward})")
                
        reward_components['path_rejoin_reward'] = path_rejoin_reward

        if self.global_path_world:
            final_goal_pos = np.array(self.global_path_world[-1])
            dist_to_final_goal = observation[2]*5.0
            
            arrival_zone_dist = 5.0 # 5 meters
            if dist_to_final_goal < arrival_zone_dist:
                w_arrival = 2.0 # A strong weight for this bonus
                arrival_reward = w_arrival / (dist_to_final_goal + 0.1)
                reward_components['arrival_reward'] = arrival_reward

        normalized_heading_error = observation[1] # This is in range [-1, 1]
        heading_error_deg = abs(normalized_heading_error * 180.0)

        w_heading_pos = 0.5 # Weight for the positive reward
        w_heading_neg = 5.0 # Weight for the exponential penalty

        heading_reward = 0.0
        # Your condition: 0 to 45 degrees is positive
        if heading_error_deg <= 45.0:
            normalized_goodness = (45.0 - heading_error_deg) / 45.0
            heading_reward = w_heading_pos * normalized_goodness
        else:
            # Your condition: 45 to 180 degrees is exponentially negative
            error_in_bad_zone = (heading_error_deg - 45.0) / 135.0
            heading_reward = -w_heading_neg * (error_in_bad_zone ** 2)

        reward_components['heading_reward'] = heading_reward

        w_deviation = 1.0
        if cross_track_error > self.path_deviation_threshold:
            deviation_penalty = 2*w_deviation * (self.path_deviation_threshold - cross_track_error)
            reward_components['deviation_penalty'] = deviation_penalty
        elif cross_track_error <= self.path_deviation_threshold:
            deviation_reward = w_deviation * (self.path_deviation_threshold - cross_track_error)
            reward_components['deviation_penalty'] = deviation_reward


        # --- B. Your Focused Proximity Penalty Logic ---
        lidar_start_idx = 6
        lidar_end_idx = lidar_start_idx + self.NUM_LIDAR_BINS # NUM_LIDAR_BINS is 32
        normalized_lidar = observation[lidar_start_idx:lidar_end_idx]
        lidar_binned_meters = normalized_lidar * self.lidar.getMaxRange()

        front_clearance = 0.0
        direct_front_bins = lidar_binned_meters[12:20] # Indices 13, 14, 15, 16, 17, 18 
        min_frontal_dist = np.min(direct_front_bins)
        # # Only activate this reward when facing a potential passage
        if min_frontal_dist < 1.75: # 1.3 meter from the head of the robot
            normalized_clearance = max(0, 1.0 - (min_frontal_dist / 1.75))
            w_centering = -2.0 
            centering_penalty = w_centering * normalized_clearance

        reward_components['front_clearance'] = front_clearance

        central_lidar_bins = lidar_binned_meters[1:-1] # All bins except the first and last
        min_frontal_range = np.min(central_lidar_bins)
        
        # Proximity Penalty
        proximity_penalty = 0.0
        proximity_warning_dist = 0.8
        if min_frontal_range < proximity_warning_dist:
            # Quadratic penalty based on how close the robot is in the frontal cone
            proximity_penalty = -8.0 * ((1/ (min_frontal_range + 1e-6)) - (1/proximity_warning_dist))**3 # Penalize for being too close to obstacles
        else:
            proximity_penalty = 0.0
        reward_components['proximity_penalty'] = proximity_penalty
        
        # --C Stagnation and Effort Penalties (Anti-Freezing/Vibrating)
        stagnation_penalty = 0.0
        if progress < 0.005 and self.global_path_world:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        if self.stagnation_counter > 30:
            stagnation_penalty = -10.0
            self.stagnation_counter = 0
            print(f"INFO: Episode {self.episode_count}: Applied stagnation penalty!")
        reward_components['stagnation_penalty'] = stagnation_penalty
        
        accumulated_rotation_penalty = 0.0
        current_omega_sign = np.sign(current_omega)
        dt = self.timestep / 1000.0 # Time per step in seconds

        # Condition for accumulating or resetting
        # Reset if: 1. Turning stops, OR 2. Turning direction reverses.
        if abs(current_omega) < 0.1 or (current_omega_sign != self.last_omega_sign and self.last_omega_sign != 0):
            # Reset the accumulator
            self.accumulated_angle = 0.0
        else:
            # If turning continues in the same direction, accumulate the rotation
            self.accumulated_angle += abs(current_omega * dt)

        # Check if the accumulated rotation exceeds the threshold
        rotation_threshold_rad = 3.5 #radians
        if self.accumulated_angle > rotation_threshold_rad:
            accumulated_rotation_penalty = -100.0
            self.accumulated_angle = 0.0
            print(f"INFO: Episode {self.episode_count}: Applied accumulated rotation penalty!")

        # Update the memory of the last turn direction for the next step
        self.last_omega_sign = current_omega_sign
        reward_components['accumulated_rotation_penalty'] = accumulated_rotation_penalty

        #  Primary Time Cost
        reward_components['time_cost'] = -0.1

        # Use normalized velocities from observation for effort penalty
        rotational_penalty = -0.1 * abs(current_omega)
        reward_components['rotational_penalty'] = rotational_penalty
        
        # Calculate Total Reward
        self.previous_position = robot_pos_np
        reward = sum(value for key, value in reward_components.items() if not key.startswith('log_'))

        # Final safety check
        if math.isnan(reward) or math.isinf(reward):
            components_str = ", ".join([f"{key}: {value:.2f}" for key, value in reward_components.items()])
            log_message = (f"CRITICAL [Ep:{self.episode_count}, Step:{self.current_step}]: "
                           f"Reward became invalid ({reward})!\n"
                           f"    COMPONENTS: {components_str}")
            print(f"\n{log_message}")
            log_debug_info(log_message)
            reward = -500
            terminated = True
        
        return reward, terminated, truncated, info
    
    def _set_robot_velocities(self, linear_vel, angular_vel):
        left_speed = (linear_vel - angular_vel * self.distance_to_center) / self.wheel_radius
        right_speed = (linear_vel + angular_vel * self.distance_to_center) / self.wheel_radius
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def _set_robot_pose(self, x, y, yaw):
        translation_field = self.robot_node.getField('translation')
        rotation_field = self.robot_node.getField('rotation')
        translation_field.setSFVec3f([x, y, 0.1])
        rotation_field.setSFRotation([0, 0, 1, yaw])
        self.robot_node.resetPhysics()
        
    def _convert_path_to_world(self, path_cells):
        world_path = []
        for cell in path_cells:
            row, col = cell
            wx = (col - ((self.c_space_grid.shape[1]-1) / 2.0)) / self.cells_per_meter
            wy = (((self.c_space_grid.shape[0]-1) / 2.0) - row) / self.cells_per_meter
            world_path.append([wx, wy])
        return world_path
    
    def _convert_cell_to_world(self, cell):
        row, col = cell
        wx = (col - ((self.c_space_grid.shape[1]-1) / 2.0)) / self.cells_per_meter
        wy = (((self.c_space_grid.shape[0]-1) / 2.0) - row) / self.cells_per_meter
        return wx, wy

    def _world_to_grid(self, world_pos):
        col = int(world_pos[0] * self.cells_per_meter + (self.c_space_grid.shape[1] - 1) / 2.0)
        row = int((self.c_space_grid.shape[0] - 1) / 2.0 - world_pos[1] * self.cells_per_meter)
        return (row, col)