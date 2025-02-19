import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from octo.model.octo_model import OctoModel
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
print(model.get_pretty_spec())
print("dataset statistics: ", model.dataset_statistics)
print("dataset statistics keys: ", model.dataset_statistics.keys())

from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

IMAGE_PATH = '/home/caleb/datasets/caleb/random_franka_photos/test_new.jpeg'
img = Image.open(IMAGE_PATH)
# img.show()
img = np.array(img)
print("img size" , img.size)

import jax 
img = img[np.newaxis,np.newaxis,...]
observation = {"image_primary": img, "timestep_pad_mask": np.array([[True]])}
# task = model.create_tasks(texts=["pick up the mustand"])
# task = model.create_tasks(texts=["pick up the red box"])
file_save_path = '/home/caleb/datasets/caleb/octo_action_taco_play_tap_table.npy'
task = model.create_tasks(texts=["tap the table"])
print("toco keys",model.dataset_statistics["taco_play"].keys())
action = model.sample_actions(
    observation, 
    task, 
    unnormalization_statistics=model.dataset_statistics["taco_play"]["action"], 
    # unnormalization_statistics=model.dataset_statistics["nyu_franka_play_dataset_converted_externally_to_rlds"]["action"], 
    rng=jax.random.PRNGKey(0)
)
print(action)

np.save(file_save_path, action)



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.load(file_save_path)

# Extract just the x, y, z coordinates for visualization
trajectory = data[0]
xyz = trajectory[:, :3]

# Compute the delta changes in position.
# diff_xyz will have shape (T-1, 3), representing the change from xyz[i] to xyz[i+1]
diff_xyz = np.diff(xyz, axis=0)

# Let's plot these delta changes as vectors.
# We'll use a 3D quiver plot where each vector starts at the previous position
# and points in the direction of the change.

# Create a new figure for a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract the starting points of each delta (previous positions)
x_start = xyz[:-1, 0]
y_start = xyz[:-1, 1]
z_start = xyz[:-1, 2]

# Extract the delta vectors (u, v, w)
u = diff_xyz[:, 0]
v = diff_xyz[:, 1]
w = diff_xyz[:, 2]

# Plot quiver arrows showing delta changes
ax.quiver(x_start, y_start, z_start, u, v, w, length=1.0, normalize=False, color='blue')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Delta Changes in Trajectory Positions')


plt.show()



