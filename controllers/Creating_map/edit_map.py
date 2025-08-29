import numpy as np
import matplotlib.pyplot as plt

# Load the map from the .npy file
factory_map = np.load('controllers/Path_planners/map_outputs/factory_map.npy')
factory_map[0:62, 0:62] = 0
# factory_map[39, 1:21] = 0
# factory_map[53:55, 19:25] = 0
# factory_map[1:39, 1] = 0
# factory_map[41:61, 1] = 0
# factory_map[1, 1:61] = 0
# factory_map[60, 1:19] = 0
# factory_map[60, 27:61] = 0
# factory_map[2:21, 60] = 0
# factory_map[35:60, 60] = 0

factory_map[0, 0:62] = 1
factory_map[61, 0:62] = 1
factory_map[0:62, 0] = 1
factory_map[0:62, 61] = 1
factory_map[53:57, 5:13] = 1
factory_map[47:51, 5:13] = 1
factory_map[40, 1:20] = 1
factory_map[46:61, 26] = 1
factory_map[27:35, 9:21] = 1
factory_map[5:21, 11:15] = 1
factory_map[5:21, 5:7] = 1
factory_map[5:21, 19:23] = 1
factory_map[5:21, 27:29] = 1
factory_map[7:9, 35:41] = 1
factory_map[7:9, 45:51] = 1
factory_map[9:49, 35:37] = 1
factory_map[9:49, 39:41] = 1
factory_map[9:49, 45:47] = 1
factory_map[9:49, 49:51] = 1
# factory_map[54:58, 54:58] = 1
factory_map[30:35, 57:61] = 1
factory_map[23:28, 57:61] = 1
factory_map[5:7, 37:39] = 1
factory_map[5:7, 47:49] = 1
# factory_map[54:58, 34:42] = 1
factory_map[57:61, 22:26] = 1
# np.save('factory_map_edited.npy', factory_map)

print("Edited map saved as factory_map_edited.npy")
# Print the shape and unique values to understand the data
# (e.g., 0 might be free space, 1 might be an obstacle)
fig, ax = plt.subplots(figsize=(10, 10)) # You can adjust the figure size
ax.imshow(factory_map, cmap='gray')

# Set the ticks to correspond to each cell's index
ax.set_xticks(np.arange(factory_map.shape[1]))
ax.set_yticks(np.arange(factory_map.shape[0]))

# Place grid lines between the cells
ax.set_xticks(np.arange(-.5, factory_map.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-.5, factory_map.shape[0], 1), minor=True)
ax.grid(which='minor', color='green', linestyle='-', linewidth=0.5)

# Rotate the x-axis labels to prevent them from overlapping
plt.xticks(rotation=90)

# Make the tick labels smaller if the map is large
ax.tick_params(axis='both', which='major', labelsize=8)

plt.show()