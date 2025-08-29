import os
import sys
import time
import csv
import math
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from typing import Callable

# --- Path finding block to locate Webots libraries ---
webots_home = os.environ.get('WEBOTS_HOME')
if webots_home:
    python_version = f"python{sys.version_info.major}{sys.version_info.minor}"
    webots_controller_path = os.path.join(webots_home, 'lib', 'controller', 'python')
    if webots_controller_path not in sys.path:
        sys.path.append(webots_controller_path)
else:
    sys.exit("ERROR: WEBOTS_HOME environment variable not set.")

# --- Import your custom environment ---
from controllers.rl_controller_env.rl_controller_env import WebotsRobotEnvironment

def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: The initial learning rate.
    :param final_value: The final learning rate.
    :return: A function that takes the remaining progress and returns the learning rate.
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1.0 (beginning) to 0.0 (end).
        """
        return final_value + progress_remaining * (initial_value - final_value)
    return func

# --- Custom Logging Callback to create step_log.csv ---
class StepDataLogger(BaseCallback):
    """A custom callback that logs detailed data at each step to a CSV file."""
    def __init__(self, log_dir: str, verbose: int = 0):
        super(StepDataLogger, self).__init__(verbose)
        self.log_dir = log_dir
        self.log_file_path = os.path.join(log_dir, "step_log.csv")
        self.log_file = None
        self.csv_writer = None

    def _on_training_start(self) -> None:
        self.log_file = open(self.log_file_path, "w", newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow([
            "total_timesteps", "episode_num", "step_in_episode", 
            "reward_for_step", "is_done", "is_truncated", 
            "robot_x", "robot_y", "robot_yaw_deg",
            "current_v", "current_omega",
            "goal_dist_obs", "goal_angle_obs",
            "target_wp_idx", "prev_dist_to_wp"
        ])
        print(f"Logging detailed step data to: {self.log_file_path}")

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        done = self.locals["dones"][0]
        info = self.locals["infos"][0]
        truncated = info.get("TimeLimit.truncated", False)
        reward = self.locals["rewards"][0]
        
        if hasattr(env, 'gps') and hasattr(env, 'imu') and hasattr(env, 'robot_node'):
            pos = env.gps.getValues()
            yaw = env.imu.getRollPitchYaw()[2]
            vel = env.robot_node.getVelocity()
            current_v = math.sqrt(vel[0]**2 + vel[1]**2)
            current_omega = vel[5]
            
            latest_obs = self.locals["new_obs"][0]
            goal_dist_obs = latest_obs[0]
            goal_angle_obs = latest_obs[1]

            self.csv_writer.writerow([
                self.num_timesteps, env.episode_count, env.current_step,
                reward, done, truncated,
                pos[0], pos[1], math.degrees(yaw),
                current_v, current_omega,
                goal_dist_obs, goal_angle_obs,
                env.target_waypoint_index, env.previous_distance_to_waypoint
            ])
        
        if done or truncated:
            self.csv_writer.writerow([]) 
        
        self.log_file.flush()
        return True

    def _on_training_end(self) -> None:
        if self.log_file is not None:
            self.log_file.close()
            print("Step data logging finished.")

# --- Configuration ---
TOTAL_TIMESTEPS = 5_000_000
EVAL_FREQ = 25000
CHECKPOINT_FREQ = 1000000

# Set path to a .zip file to resume training, or set to None to start from scratch.
LOAD_FROM_MODEL = None
# Use different directories for continued runs to not overwrite previous results
if LOAD_FROM_MODEL:
    LOG_DIR = "logs_continued/"
    MODEL_DIR = "models_continued/"
else:
    LOG_DIR = "logs_initial/"
    MODEL_DIR = "models_initial/"

FAILURE_LOG_PATH = None

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


if __name__ == "__main__":
    # --- 1. Create a SINGLE Environment Instance  ---
    env = Monitor(WebotsRobotEnvironment(render_mode='none', failure_log_path=FAILURE_LOG_PATH))

    unwrapped_env = env.unwrapped 

    # STAGE 1: Learn basic path following
    unwrapped_env.static_obstacle_phase = True
    unwrapped_env.dynamic_obstacle_phase = False
    vec_env = DummyVecEnv([lambda: env])
    train_env = VecFrameStack(vec_env, n_stack=4)  # Use VecFrameStack for compatibility
    
    # --- 2. Set up ALL Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ, save_path=MODEL_DIR, name_prefix="ppo_model"
    )
    
    eval_callback = EvalCallback(
        train_env, # Use the same env instance
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )

    # Instantiate  custom logger
    step_logger = StepDataLogger(log_dir=LOG_DIR)

    # --- 3. Create or LOAD the Model ---
    if LOAD_FROM_MODEL and os.path.exists(LOAD_FROM_MODEL):
        print(f"\n--- Loading pre-trained model from: {LOAD_FROM_MODEL} ---")
        model = PPO.load(
            LOAD_FROM_MODEL,
            env=train_env,
            tensorboard_log=LOG_DIR
        )
        print("Model loaded successfully. Resuming training.")
    else:
        print("\n--- No pre-trained model found or specified. Training new model from scratch. ---")
        policy_kwargs = dict(
            activation_fn=nn.ReLU, # Use ReLU instead of the default Tanh
            net_arch=dict(
                pi=[512, 256, 128],  # Actor (policy) network
                vf=[512, 256, 128]   # Critic (value) network
        )
    )
        # lr_schedule = linear_schedule(3e-4, 1e-5) # Start at 0.0003, end at 0.00001
        model = PPO(
            "MlpPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1,
            tensorboard_log=LOG_DIR, device="cpu", n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            learning_rate=3e-4
        )
    
    # --- 4. Start Training ---
    print(f"--- Starting PPO Training ---")
    print(f"Logs will be saved in: '{LOG_DIR}'")
    print(f"Models will be saved in: '{MODEL_DIR}'")
    
    start_time = time.time()

    try:
        # Pass ALL callbacks to the learn method
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback, step_logger],
            progress_bar=True,
            # Set reset_num_timesteps to False if you are resuming training
            reset_num_timesteps=False if LOAD_FROM_MODEL else True
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- 5. Save and Clean Up ---
        model.save(f"{MODEL_DIR}/ppo_model_final_{TOTAL_TIMESTEPS}_steps")
        train_env.close()
        print("\nTraining script finished or was interrupted.")
        print(f"The best performing model (if any) was saved as 'best_model.zip' in '{MODEL_DIR}'.")
        print(f"Detailed step data was logged to '{LOG_DIR}/step_log.csv'.")
