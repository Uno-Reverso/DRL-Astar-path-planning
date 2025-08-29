import os
import sys
import numpy as np
import csv
import datetime
import json

# --- Path finding block to locate Webots libraries ---
webots_home = os.environ.get('WEBOTS_HOME')
if webots_home:
    python_version = f"python{sys.version_info.major}{sys.version_info.minor}"
    webots_controller_path = os.path.join(webots_home, 'lib', 'controller', 'python')
    if webots_controller_path not in sys.path:
        sys.path.append(webots_controller_path)
else:
    sys.exit("ERROR: WEBOTS_HOME environment variable not set.")

# --- Import SB3 and Env ---
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import matplotlib.pyplot as plt

from controllers.rl_controller_env.rl_controller_env import WebotsRobotEnvironment
from controllers.Gobal_planner import a_star_planner

# --- Configuration ---
MODEL_PATH = "models_initial_With_frontLidar/best_model.zip"
FAILURE_LOG_PATH = None
NUM_EPISODES_PER_CONDITION =  1
RENDER_MODE = 'none'  # 'plot' to see live plots, 'none' for no rendering

RUN_PATH_FOLLOWING_TEST = True
RUN_OBSTACLE_TEST = False

LOG_DIR = "test_logs/"
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
FAILURE_LOG_FILE = os.path.join(LOG_DIR, f"failures_{timestamp}.csv")

def convert_numpy_types(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
    if isinstance(obj, tuple): return tuple(convert_numpy_types(i) for i in obj)
    return obj

def run_evaluation_condition(model, env, condition_name, has_obstacles, num_episodes):
    print(f"\n--- Running Condition: {condition_name} for {num_episodes} episodes ---")

    all_trajectories = []

    underlying_env = env.envs[0].env
    underlying_env.static_obstacle_phase = has_obstacles

    outcomes = {'success': 0, 'collision': 0, 'timeout': 0}
    successful_runs_metrics = {
        'trajectory_efficiency': [], 'normalised_time': [], 'focused_clearance': [],
        'trajectory_smoothness': [], 'reaction_time': [], 'cross_track_error': []
    }
    overall_completion_ratios = []
    post_obstacle_completion_ratios = []

    episodes_completed = 0
    obs = env.reset()
    
    while episodes_completed < num_episodes:
        spawned_obstacle_world_pos = None
        obstacle_dist_along_path = 0.0
        obstacle_segment_index = -1
        current_trajectory = []

        if has_obstacles and underlying_env.current_obstacle_params:
            path_cells = a_star_planner.find_path(underlying_env.c_space_grid, underlying_env.current_start_cell, underlying_env.current_goal_cell, underlying_env.dist_transform)
            if path_cells and len(path_cells) > underlying_env.current_obstacle_params[0][0]:
                params = underlying_env.current_obstacle_params[0]
                center_node_index, side = params
                obstacle_segment_index = center_node_index
                obstacle_dist_along_path = underlying_env.cumulative_segment_lengths[center_node_index]
                p1 = np.array(underlying_env._convert_cell_to_world(path_cells[center_node_index - 1]))
                p2 = np.array(underlying_env._convert_cell_to_world(path_cells[center_node_index + 1]))
                path_dir_vec = p2 - p1
                perp_vec = np.array([-path_dir_vec[1], path_dir_vec[0]])
                perp_vec_norm = perp_vec / (np.linalg.norm(perp_vec) + 1e-6)
                center_node_world = np.array(underlying_env._convert_cell_to_world(path_cells[center_node_index]))
                spawned_obstacle_world_pos = center_node_world + side * 0.5 * perp_vec_norm

        total_distance_traveled = 0.0
        last_pos = underlying_env.gps.getValues()[:2]
        last_action = np.zeros(2)
        clearance_readings, jerk_values, cte_values = [], [], []
        t_perceive, t_respond = -1, -1
        was_turning_at_perception = False

        while True:
            action, _states = model.predict(obs, deterministic=True)
            current_action = action[0]
            jerk = np.linalg.norm(current_action - last_action)
            jerk_values.append(jerk)
            last_action = current_action
            
            current_pos = underlying_env.gps.getValues()[:2]
            current_trajectory.append(current_pos)
            angular_vel = current_action[1] * underlying_env.max_angular_velocity

            if has_obstacles and spawned_obstacle_world_pos is not None:
                if t_perceive == -1:
                    dist_to_obstacle = np.linalg.norm(np.array(current_pos) - spawned_obstacle_world_pos)
                    if dist_to_obstacle < 2.5:
                        t_perceive = underlying_env.current_step
                        was_turning_at_perception = abs(angular_vel) > 0.3
                
                if t_perceive != -1 and t_respond == -1 and not was_turning_at_perception:
                    if abs(angular_vel) > 0.5:
                        t_respond = underlying_env.current_step
            
            obs, reward, done, infos = env.step(action)
            
            total_distance_traveled += np.linalg.norm(np.array(current_pos) - np.array(last_pos))
            last_pos = current_pos
            
            lidar_obs = obs[0][6:]
            min_central_clearance = np.min(lidar_obs[14:18]) * underlying_env.lidar.getMaxRange()
            clearance_readings.append(min_central_clearance)

            if not has_obstacles:
                proj, _, _ = underlying_env._get_path_projection_info(np.array(current_pos))
                cte_values.append(np.linalg.norm(np.array(current_pos) - proj))

            if done:
                episodes_completed += 1
                info = infos[0]

                all_trajectories.append({
                    "episode_num": episodes_completed,
                    "outcome": info.get('termination_reason', 'unknown'),
                    "trajectory": convert_numpy_types(current_trajectory) # Use your converter
                })

                termination_reason = info.get('termination_reason', 'unknown')
                a_star_path_length = underlying_env.cumulative_segment_lengths[-1] + underlying_env.segment_lengths[-1] if underlying_env.segment_lengths.size > 0 else 0

                if termination_reason == 'goal_reached':
                    outcomes['success'] += 1
                    if has_obstacles:
                        post_obstacle_completion_ratios.append(1.0)
                    else:
                        overall_completion_ratios.append(1.0)

                    if a_star_path_length > 0:
                        simulated_time = info['episode']['l'] * (underlying_env.timestep / 1000.0)
                        successful_runs_metrics['normalised_time'].append(simulated_time / a_star_path_length)
                    
                    successful_runs_metrics['focused_clearance'].append(np.mean(clearance_readings))
                    successful_runs_metrics['trajectory_smoothness'].append(np.mean(jerk_values))

                    if has_obstacles:
                        if t_perceive != -1 and t_respond != -1:
                            reaction_time_sec = (t_respond - t_perceive) * (underlying_env.timestep / 1000.0)
                            successful_runs_metrics['reaction_time'].append(reaction_time_sec)
                    else:
                        if total_distance_traveled > 0.1 and a_star_path_length > 0:
                            successful_runs_metrics['trajectory_efficiency'].append(a_star_path_length / total_distance_traveled)
                        successful_runs_metrics['cross_track_error'].append(np.mean(cte_values))

                else:
                    if termination_reason == 'collision': outcomes['collision'] += 1
                    elif termination_reason == 'timeout': outcomes['timeout'] += 1
                    
                    if a_star_path_length > 0:
                        _proj, dist_along, final_robot_seg_idx = underlying_env._get_path_projection_info(np.array(last_pos))
                        
                        if has_obstacles:
                            if final_robot_seg_idx > obstacle_segment_index:
                                progress_after_obstacle = dist_along - obstacle_dist_along_path
                                remaining_path_length = a_star_path_length - obstacle_dist_along_path
                                survival_ratio = max(0, progress_after_obstacle / remaining_path_length) if remaining_path_length > 0 else 0
                                post_obstacle_completion_ratios.append(survival_ratio)
                            else:
                                post_obstacle_completion_ratios.append(0.0)
                        else:
                            overall_completion_ratios.append(dist_along / a_star_path_length)

                    with open(FAILURE_LOG_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        obstacle_params = json.dumps(convert_numpy_types(underlying_env.current_obstacle_params))
                        writer.writerow([
                            episodes_completed, condition_name, termination_reason, info['episode']['l'],
                            underlying_env.current_start_yaw,
                            underlying_env.current_start_cell[0], underlying_env.current_start_cell[1],
                            underlying_env.current_goal_cell[0], underlying_env.current_goal_cell[1],
                            obstacle_params, last_pos[0], last_pos[1]
                        ])
                
                print(f"  - Episode {episodes_completed}/{num_episodes} complete. Outcome: {termination_reason}")
                break
    
    trajectory_log_file = os.path.join(LOG_DIR, f"trajectories_{condition_name.replace(' ', '_')}_{timestamp}.json")
    try:
        with open(trajectory_log_file, 'w') as f:
            json.dump(all_trajectories, f, indent=2)
        print(f"Saved {len(all_trajectories)} trajectories to: {trajectory_log_file}")
    except Exception as e:
        print(f"Error saving trajectories: {e}")

    summary = {}
    total_runs = sum(outcomes.values())
    if total_runs > 0:
        for key, value in outcomes.items(): summary[f'{key}_rate'] = (value / total_runs) * 100
    if overall_completion_ratios:
        summary['overall_completion_mean'] = np.mean(overall_completion_ratios) * 100
        summary['overall_completion_std'] = np.std(overall_completion_ratios) * 100
    if post_obstacle_completion_ratios:
        summary['post_obstacle_completion_mean'] = np.mean(post_obstacle_completion_ratios) * 100
        summary['post_obstacle_completion_std'] = np.std(post_obstacle_completion_ratios) * 100
    for metric, data in successful_runs_metrics.items():
        if data:
            summary[f'{metric}_mean'] = np.mean(data)
            summary[f'{metric}_std'] = np.std(data)
    return summary

def print_summary_table(summary_dict, condition_name):
    print(f"\n--- Summary for: {condition_name} ---")
    print("-" * 50)
    print(f"{'Metric':<28} | {'Mean':<10} | {'Std Dev':<10}")
    print("-" * 50)
    def print_row(name, key_base, unit):
        mean_key, std_key = f"{key_base}_mean", f"{key_base}_std"
        if mean_key in summary_dict:
            mean_val = f"{summary_dict[mean_key]:.2f}"
            std_val = f"{summary_dict.get(std_key, 0):.2f}"
            print(f"{name:<28} | {mean_val:<10} | {std_val:<10} {unit}")
        else:
            print(f"{name:<28} | {'N/A':<10} | {'N/A':<10}")

    print(f"{'Success Rate (%)':<28} | {summary_dict.get('success_rate', 0):.2f}")
    print(f"{'Collision Rate (%)':<28} | {summary_dict.get('collision_rate', 0):.2f}")
    print(f"{'Timeout Rate (%)':<28} | {summary_dict.get('timeout_rate', 0):.2f}")
    print("-" * 50)
    
    print_row("Overall Path Completion (%)", "overall_completion", "")
    print_row("Post-Obstacle Completion (%)", "post_obstacle_completion", "")
    print_row("Trajectory Efficiency", "trajectory_efficiency", "")
    print_row("Cross-Track Error (CTE)", "cross_track_error", "m")
    print_row("Normalised Time", "normalised_time", "s/m")
    print_row("Focused Clearance", "focused_clearance", "m")
    print_row("Trajectory Smoothness", "trajectory_smoothness", "(Jerk)")
    print_row("Reaction Time", "reaction_time", "s")
    print("-" * 50)

if __name__ == "__main__":
    with open(FAILURE_LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode_num", "condition", "outcome", "steps_taken", "start_yaw_rad", "start_cell_row", "start_cell_col",
            "goal_cell_row", "goal_cell_col", "obstacle_params_json", "final_x", "final_y"
        ])
    print(f"Logging failure scenarios to: {FAILURE_LOG_FILE}")

    try:
        print(f"Loading trained model from: {MODEL_PATH}")
        model = PPO.load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Error loading model: {e}"); exit()

    print("\n\n" + "="*50)
    print("           STARTING EVALUATION RUN")
    print("="*50)
    
    # --- FIX: Create a completely separate environment for each test run ---
    if RUN_PATH_FOLLOWING_TEST:
        print("\nCreating new environment for Path Following test...")
        env_pf = None
        try:
            env_pf = VecFrameStack(DummyVecEnv([lambda: Monitor(WebotsRobotEnvironment(render_mode=RENDER_MODE, failure_log_path=FAILURE_LOG_PATH, is_testing=True))]), n_stack=4)
            model.set_env(env_pf)
            path_following_summary = run_evaluation_condition(
                model, env_pf, "Path Following", has_obstacles=False, num_episodes=NUM_EPISODES_PER_CONDITION
            )
        finally:
            plt.ioff()
            plt.show()
            if env_pf is not None:
                env_pf.close()
    
    if RUN_OBSTACLE_TEST:
        print("\nCreating new environment for Obstacle Avoidance test...")
        env_oa = None
        try:
            env_oa = VecFrameStack(DummyVecEnv([lambda: Monitor(WebotsRobotEnvironment(render_mode=RENDER_MODE, failure_log_path=FAILURE_LOG_PATH, is_testing=True))]), n_stack=4)
            model.set_env(env_oa)
            obstacle_avoidance_summary = run_evaluation_condition(
                model, env_oa, "Obstacle Avoidance", has_obstacles=True, num_episodes=NUM_EPISODES_PER_CONDITION
            )
        finally:
            plt.ioff()
            plt.show()
            if env_oa is not None:
                env_oa.close()

    print("\n\n" + "="*50)
    print("           FINAL EVALUATION RESULTS")
    print("="*50)
    if RUN_PATH_FOLLOWING_TEST:
        print_summary_table(path_following_summary, "Path Following (No Obstacles)")
    if RUN_OBSTACLE_TEST:
        print_summary_table(obstacle_avoidance_summary, "Obstacle Avoidance")
    
    print("\nEvaluation script finished.")
