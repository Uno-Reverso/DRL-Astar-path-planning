import numpy as np
import matplotlib.pyplot as plt
import heapq 
import random
import cv2

class Node:
    def __init__(self, position, parent=None, direction=(0,0)):
        self.parent = parent
        self.position = position
        self.direction = direction # Direction of the move that reached this node
        self.g, self.h, self.f = 0, 0, 0
    def __eq__(self, other): return self.position == other.position
    def __lt__(self, other): return self.f < other.f
    def __hash__(self): return hash(self.position)

# --- 2. The A* Function (UPDATED with Turn Penalty Logic) ---
def find_path(grid, start, end, dist_transform):
    # --- Configuration for the planner ---
    # A higher value makes the planner prefer straighter paths more strongly.
    TURN_PENALTY = 2.0 
    WALL_AVOIDANCE_STRENGTH = 50.0 
    # ------------------------------------

    start_node, end_node = Node(start), Node(end)
    open_set, closed_set = [], set()
    heapq.heappush(open_set, start_node)
    height, width = grid.shape
    
    while open_set:
        current_node = heapq.heappop(open_set)
        closed_set.add(current_node)

        if current_node == end_node:
            path = []
            current = current_node
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        # Iterate through all 8 possible neighbor moves
        for move_x, move_y, move_cost in [(-1,-1,1.414), (-1,1,1.414), (1,-1,1.414), (1,1,1.414), 
                                          (0,-1,1.0), (0,1,1.0), (-1,0,1.0), (1,0,1.0)]:
            
            node_position = (current_node.position[0] + move_x, current_node.position[1] + move_y)

            # Check if the move is valid
            if not (0 <= node_position[0] < height and 0 <= node_position[1] < width) or \
               grid[node_position[0]][node_position[1]] != 0:
                continue

            # Create the new node, storing the direction of this move
            move_direction = (move_x, move_y)
            new_node = Node(node_position, current_node, move_direction)
            if new_node in closed_set: continue

            # --- NEW: Calculate the turn penalty ---
            turn_cost = 0
            # We check if the direction to this new node is different from the direction that led to the current node
            if current_node.parent is not None and new_node.direction != current_node.direction:
                turn_cost = TURN_PENALTY
            # -------------------------------------

            # Calculate the total cost (g) to reach this new node
            new_g = current_node.g + move_cost + turn_cost
            
            # Heuristic and Wall Avoidance (unchanged)
            h = np.sqrt(((new_node.position[0] - end_node.position[0])**2) + \
                        ((new_node.position[1] - end_node.position[1])**2))
            dist_to_wall = dist_transform[new_node.position[0]][new_node.position[1]]
            penalty = WALL_AVOIDANCE_STRENGTH / (dist_to_wall + 1e-6)
            
            # Final f-score
            new_f = new_g + h + penalty

            # Check if this path to the node is better than any existing one
            is_in_open_set = False
            for open_node in open_set:
                if new_node == open_node and new_g >= open_node.g:
                    is_in_open_set = True
                    break
            
            if not is_in_open_set:
                new_node.g, new_node.h, new_node.f = new_g, h, new_f
                heapq.heappush(open_set, new_node)
    return None

# --- HELPER FUNCTION (Unchanged) ---
def find_random_free_space(grid):
    max_attempts = 10000
    height, width = grid.shape
    for _ in range(max_attempts):
        row, col = random.randint(0, height - 1), random.randint(0, width - 1)
        if grid[row][col] == 0: return (row, col)
    raise RuntimeError("Could not find a random free space on the map.")

# --- Helper and Visualization Functions (Unchanged) ---
def create_c_space(original_grid):
    print("Inflating obstacles by 1 cell to create safe C-space...")
    height, width = original_grid.shape
    c_space_grid = np.copy(original_grid)
    obstacle_indices = np.argwhere(original_grid == 1)
    for r_obs, c_obs in obstacle_indices:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                r_neighbor, c_neighbor = r_obs + dr, c_obs + dc
                if 0 <= r_neighbor < height and 0 <= c_neighbor < width:
                    c_space_grid[r_neighbor, c_neighbor] = 1
    return c_space_grid

# --- 4. Test and Visualization Block (UPDATED) ---
if __name__ == '__main__':
    print("Running A* Planner Test with C-Space and Geometric P/W Smoothing...")

    try:
        # This path must be correct for your system
        map_path = 'controllers/Path_planners/map_outputs/factory_map_edited.npy'
        original_grid = np.load(map_path)
    except FileNotFoundError:
        print(f"ERROR: Map file not found at '{map_path}'.")
        exit()

    # --- NEW: Add a check for map dimensions ---
    height, width = original_grid.shape
    if height > 100 or width > 100:
        print("\n" + "="*60)
        print("!! ERROR: MAP IS TOO LARGE !!")
        print(f"The loaded map has dimensions {height}x{width}, but it should be much smaller (e.g., 45x45).")
        print("This means you are loading an old, high-resolution map file.")
        print("\nSOLUTION:")
        print("1. Go to your 'map_outputs' folder.")
        print("2. DELETE the 'factory_map.npy' file.")
        print("3. Run the newest 'map_generator.py' script again to create the correct, smaller map.")
        print("="*60 + "\n")
        exit()
    # --- END OF NEW CHECK ---
    
    c_space_grid = create_c_space(original_grid)
    # c_space_grid = original_grid.copy()  # For testing, we use the original grid directly
    
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(c_space_grid), cv2.DIST_L2, 5)
    
    try:
        start_point = find_random_free_space(c_space_grid)
        end_point = find_random_free_space(c_space_grid)
        # start_point = (55,3)
        # end_point = (25,3)    
    except RuntimeError as e:
        print(f"ERROR: {e}")
        exit()
    
    raw_path = find_path(c_space_grid, start_point, end_point, dist_transform)  
    
    if raw_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle('A* Path Planning Comparison', fontsize=16)

        ax1.imshow(original_grid, cmap='Greys', origin='lower')
        ax1.set_title("Path on Original Map")
        
        ax2.imshow(c_space_grid, cmap='Greys', origin='lower')
        ax2.set_title("Path on Inflated C-Space Map")
        
        raw_y, raw_x = zip(*raw_path)
        for ax in [ax1, ax2]:
            ax.plot(raw_x, raw_y, 'r-', label='Raw Path')
            ax.plot(start_point[1], start_point[0], 'go', markersize=5, label='Start')
            ax.plot(end_point[1], end_point[0], 'ro', markersize=5, label='End')
            
            # --- NEW: More prominent grid lines for better visibility ---
            # Set major ticks for the grid lines
            ax.set_xticks(np.arange(-0.5, original_grid.shape[1], 1), minor=False)
            ax.set_yticks(np.arange(-0.5, original_grid.shape[0], 1), minor=False)
            # Make grid lines thicker and more visible
            ax.grid(which='major', color='red', linestyle=':', linewidth=0.7, alpha=0.6)
            # Hide the tick labels for a cleaner look
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            ax.legend()

        plt.show()
    else:
        print("No path found.")
