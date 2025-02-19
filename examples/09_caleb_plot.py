# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Load the data from the .npy file
# # Replace 'path/to/your_data.npy' with the actual path to your .npy file
# # data = np.load('path/to/your_data.npy')
# data = np.load('/home/caleb/datasets/caleb/octo_action_taco_play.npy')


# # The data is assumed to have shape (N, T, D).
# # For this example, we assume N=1 (a single trajectory), T=4 (timesteps), D=7 (dimensions).
# #
# # trajectory: shape (T, D)
# trajectory = data[0]

# # Extract just the x, y, z coordinates
# xyz = trajectory[:, :3]

# # Compute the delta changes in position.
# # diff_xyz will have shape (T-1, 3), representing the change from xyz[i] to xyz[i+1]
# diff_xyz = np.diff(xyz, axis=0)

# # Let's plot these delta changes as vectors.
# # We'll use a 3D quiver plot where each vector starts at the previous position
# # and points in the direction of the change.

# # Create a new figure for a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Extract the starting points of each delta (previous positions)
# x_start = xyz[:-1, 0]
# y_start = xyz[:-1, 1]
# z_start = xyz[:-1, 2]

# # Extract the delta vectors (u, v, w)
# u = diff_xyz[:, 0]
# v = diff_xyz[:, 1]
# w = diff_xyz[:, 2]

# # Plot quiver arrows showing delta changes
# ax.quiver(x_start, y_start, z_start, u, v, w, length=1.0, normalize=False, color='blue')

# # Add labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Delta Changes in Trajectory Positions')

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data from the .npy file
# Replace 'path/to/your_data.npy' with the actual path to your .npy file
# data = np.load('/home/caleb/datasets/caleb/octo_action_taco_play.npy')
# data = np.load('/home/caleb/datasets/caleb/octo_action_taco_play_red_box.npy')
data = np.load('/home/caleb/datasets/caleb/octo_action_taco_play_wood_block.npy')


# The data is assumed to have shape (N, T, D), where:
# N = number of trajectories (often 1 if just one trajectory)
# T = number of timesteps (here 4)
# D = dimensions of each timestep data (here 7)
#
# We assume the first three entries in each timestep correspond to (x, y, z) coordinates.
# For example, data might look like: data[0] = [[x0, y0, z0, ...], [x1, y1, z1, ...], ...]
#
# Extract just the x, y, z coordinates for visualization
trajectory = data[0, :, :3]

# Create a new figure for a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Unpack the trajectory data
x = trajectory[:, 0]
y = trajectory[:, 1]
z = trajectory[:, 2]

# Plot the trajectory
ax.plot(x, y, z, marker='o', linestyle='-', color='blue', label='Trajectory')

# Label the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add a title
ax.set_title('3D Trajectory Visualization')

# Optionally, add a legend
ax.legend()

# Show the plot
plt.show()
