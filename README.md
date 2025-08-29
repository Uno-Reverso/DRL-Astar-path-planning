# Quan Dissertation: RL Robot Navigation

This repository contains the code, environments, and documentation for Quan Vu's dissertation on reinforcement learning for robot navigation in Webots.

---

## 📁 Project Structure

```
Quan_dissertation/
├── advanced_test.py                # Evaluation script for trained models
├── train.py                        # Training script for RL agent
├── map_plotting.py, plot.py        # Visualization utilities
├── controllers/                    # Custom Webots controllers and planners
│   ├── rl_controller_env/
│   │   └── rl_controller_env.py    # Custom RL environment for Webots
│   ├── Gobal_planner/
│   │   └── a_star_planner.py       # A* path planner
│   └── Creating_map/
│       └── edit_map.py             # Map editing tools
├── models_initial/                 # Saved models
├── logs_initial/                   # Training logs and evaluation results
├── worlds/                         # Webots world files
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Quan_dissertation.git
cd Quan_dissertation
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Webots

- Download and install [Webots](https://cyberbotics.com/).
- Set the `WEBOTS_HOME` environment variable:
    ```bash
    export WEBOTS_HOME=/path/to/webots
    ```
- The robot controller must be set to <extern>
---

## 🏋️‍♂️ Training

To train a new RL model:

```bash
python train.py
```

- Training logs will be saved in `logs_initial/`
- Models will be saved in `models_initial/`

---

## 🧪 Evaluation

To evaluate a trained model:

```bash
python advanced_test.py
```
Configurations:
- MODEL_PATH                                        - Copy the directory to your model (.zip)
- FAILURE_LOG_PATH                                  - Copy the directory of the failue csv file test_log if you want to re-do scenarios
- NUM_EPISODES_PER_CONDITION                        - Choose the number episode you want to run
- RENDER_MODE =                                     -'none'  # 'plot' to see live plots, 'none' for no rendering
- RUN_PATH_FOLLOWING_TEST and UN_OBSTACLE_TEST      - only choose one
  
Failures and trajectory logs will be saved in `test_logs/`

---

## 🗺️ Map Editing & Planning

- Edit maps:  
  ```bash
  python controllers/Creating_map/edit_map.py
  ```
- Path planning utilities are in `controllers/Gobal_planner/a_star_planner.py`

---

## 📄 License

This project is licensed under the MIT License.

---

## 📬 Contact

For questions, contact Quan Vu at [a.q.vu@edu.salford.ac.uk].
