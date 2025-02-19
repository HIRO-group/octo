# import os
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# import cv2
# import time
# import numpy as np
# from PIL import Image
# import jax
# from octo.model.octo_model import OctoModel
# import paramiko
# import matplotlib.pyplot as plt
# from pynput import keyboard

# # Remote SSH details from environment variables
# host = os.getenv("SSH_HOST", "127.0.0.1")  # Default to localhost if not set
# user = os.getenv("SSH_USER", "user")
# password = os.getenv("SSH_PASSWORD", "password")
# remote_path = "/home/caleb/robochem_steps/octo_action.txt"

# # Set up SSH & SFTP connection once outside the loop for efficiency
# ssh = paramiko.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect(host, username=user, password=password)
# sftp = ssh.open_sftp()

# # Load the pretrained model once
# model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")

# # Create a fixed task (adjust the text prompt as needed)
# task_text = "reach up in the air"
# task = model.create_tasks(texts=[task_text])

# # Open the camera
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise Exception("Could not open camera.")

# # Initialize cropping parameters
# crop_width = 256
# crop_height = 256

# # Function to handle key press for updating the task input
# def on_press(key):
#     global task_text, task
#     try:
#         if key.char == 'g':
#             print("\n'g' key detected. Enter a new task input:")
#             task_text = input("New task: ")
#             task = model.create_tasks(texts=[task_text])
#             print(f"Task updated to: {task_text}")
#     except AttributeError:
#         pass

# # Start keyboard listener
# listener = keyboard.Listener(on_press=on_press)
# listener.start()

# try:
#     fig, ax = plt.subplots()
#     plt.subplots_adjust(bottom=0.2)

#     # Initialize sliders for cropping X and Y
#     ax_slider_x = plt.axes([0.25, 0.1, 0.65, 0.03])
#     ax_slider_y = plt.axes([0.25, 0.05, 0.65, 0.03])
#     slider_x = plt.Slider(ax_slider_x, 'Crop X', 0, 1, valinit=0, valstep=0.01)
#     slider_y = plt.Slider(ax_slider_y, 'Crop Y', 0, 1, valinit=0, valstep=0.01)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to read from camera. Exiting.")
#             break

#         # Get cropping parameters from sliders
#         crop_x = int(slider_x.val * (frame.shape[1] - crop_width))
#         crop_y = int(slider_y.val * (frame.shape[0] - crop_height))

#         # Crop the frame safely
#         cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

#         # Resize and convert to RGB
#         resized = cv2.resize(cropped_frame, (256, 256))
#         resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

#         # Display the cropped frame and update the title with the current task text
#         ax.clear()
#         ax.imshow(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
#         ax.set_title(f"Current Task: {task_text}")
#         plt.draw()
#         plt.pause(0.05)

#         # Prepare input for the model
#         img = resized_rgb[np.newaxis, np.newaxis, ...]
#         observation = {
#             "image_primary": img,
#             "timestep_pad_mask": np.array([[True]])
#         }

#         # Model inference
#         action = model.sample_actions(
#             observation,
#             task,
#             unnormalization_statistics=model.dataset_statistics["taco_play"]["action"],
#             rng=jax.random.PRNGKey(0)
#         )

#         # Write action to a file
#         local_action_file = "action.txt"
#         with open(local_action_file, "w") as f:
#             line = ",".join(f"{val:.6f}" for val in action[0][0])
#             f.write(line + "\n")

#         # Upload the file
#         sftp.put(local_action_file, remote_path)

#         # Wait to reduce loop intensity
#         time.sleep(0.05)

# except KeyboardInterrupt:
#     print("Interrupted by user.")
# finally:
#     # Cleanup resources
#     cap.release()
#     sftp.close()
#     ssh.close()
#     plt.close(fig)
#     listener.stop()

# import os
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# import cv2
# import time
# import numpy as np
# from PIL import Image
# import jax
# from octo.model.octo_model import OctoModel
# import paramiko
# import matplotlib.pyplot as plt
# import keyboard

# # Remote SSH details from environment variables
# host = os.getenv("SSH_HOST", "127.0.0.1")  # Default to localhost if not set
# user = os.getenv("SSH_USER", "user")
# password = os.getenv("SSH_PASSWORD", "password")
# remote_path = "/home/caleb/robochem_steps/octo_action.txt"

# # Set up SSH & SFTP connection once outside the loop for efficiency
# ssh = paramiko.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect(host, username=user, password=password)
# sftp = ssh.open_sftp()

# # Load the pretrained model once
# model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")

# # Create a fixed task (adjust the text prompt as needed)
# task_text = "reach up in the air"
# task = model.create_tasks(texts=[task_text])

# # Open the camera
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise Exception("Could not open camera.")

# # Initialize cropping parameters
# crop_width = 256
# crop_height = 256

# try:
#     fig, ax = plt.subplots()
#     plt.subplots_adjust(bottom=0.2)

#     # Initialize sliders for cropping X and Y
#     ax_slider_x = plt.axes([0.25, 0.1, 0.65, 0.03])
#     ax_slider_y = plt.axes([0.25, 0.05, 0.65, 0.03])
#     slider_x = plt.Slider(ax_slider_x, 'Crop X', 0, 1, valinit=0, valstep=0.01)
#     slider_y = plt.Slider(ax_slider_y, 'Crop Y', 0, 1, valinit=0, valstep=0.01)

#     while True:
#         # Check if 'g' key is pressed for task update
#         if keyboard.is_pressed('g'):
#             print("\n'g' key detected. Enter a new task input:")
#             task_text = input("New task: ")
#             task = model.create_tasks(texts=[task_text])
#             print(f"Task updated to: {task_text}")

#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to read from camera. Exiting.")
#             break

#         # Get cropping parameters from sliders
#         crop_x = int(slider_x.val * (frame.shape[1] - crop_width))
#         crop_y = int(slider_y.val * (frame.shape[0] - crop_height))

#         # Crop the frame safely
#         cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

#         # Resize and convert to RGB
#         resized = cv2.resize(cropped_frame, (256, 256))
#         resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

#         # Display the cropped frame and update the title with the current task text
#         ax.clear()
#         ax.imshow(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
#         ax.set_title(f"Current Task: {task_text}")
#         plt.draw()
#         plt.pause(0.05)

#         # Prepare input for the model
#         img = resized_rgb[np.newaxis, np.newaxis, ...]
#         observation = {
#             "image_primary": img,
#             "timestep_pad_mask": np.array([[True]])
#         }

#         # Model inference
#         action = model.sample_actions(
#             observation,
#             task,
#             unnormalization_statistics=model.dataset_statistics["taco_play"]["action"],
#             rng=jax.random.PRNGKey(0)
#         )

#         # Write action to a file
#         local_action_file = "action.txt"
#         with open(local_action_file, "w") as f:
#             line = ",".join(f"{val:.6f}" for val in action[0][0])
#             f.write(line + "\n")

#         # Upload the file
#         sftp.put(local_action_file, remote_path)

#         # Wait to reduce loop intensity
#         time.sleep(0.05)

# except KeyboardInterrupt:
#     print("Interrupted by user.")
# finally:
#     # Cleanup resources
#     cap.release()
#     sftp.close()
#     ssh.close()
#     plt.close(fig)

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import cv2
import time
import numpy as np
from PIL import Image
import jax
from octo.model.octo_model import OctoModel
import paramiko
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import traceback

# Remote SSH details from environment variables
host = os.getenv("SSH_HOST", "127.0.0.1")  # Default to localhost if not set
user = os.getenv("SSH_USER", "user")
password = os.getenv("SSH_PASSWORD", "password")
remote_path = "/home/caleb/robochem_steps/octo_action.txt"

# Set up SSH & SFTP connection once outside the loop for efficiency
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, username=user, password=password)
sftp = ssh.open_sftp()

# Load the pretrained model once
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")

# Create a fixed task (adjust the text prompt as needed)
task_text = "reach up in the air"
task = model.create_tasks(texts=[task_text])

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open camera.")

# Initialize cropping parameters
crop_width = 256
crop_height = 256

# Callback function to update task input
def update_task_input(event):
    global task_text, task
    print("Button clicked. Enter new task:")
    task_text = input("New task: ")
    task = model.create_tasks(texts=[task_text])
    print(f"Task updated to: {task_text}")

try:
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)

    # Initialize sliders for cropping
    ax_slider_x = plt.axes([0.25, 0.2, 0.65, 0.03])
    ax_slider_y = plt.axes([0.25, 0.15, 0.65, 0.03])
    slider_x = Slider(ax_slider_x, 'Crop X', 0, 1, valinit=0.87, valstep=0.01)
    slider_y = Slider(ax_slider_y, 'Crop Y', 0, 1, valinit=0.7, valstep=0.01)

    # Add a button for task input update
    ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
    button = Button(ax_button, 'Update Task')
    button.on_clicked(update_task_input)

    rng_key = jax.random.PRNGKey(0)

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera. Exiting.")
                break

            # Get cropping parameters from sliders
            crop_x = int(slider_x.val * (frame.shape[1] - crop_width))
            crop_y = int(slider_y.val * (frame.shape[0] - crop_height))

            # Crop the frame safely
            cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

            # Resize and convert to RGB
            resized = cv2.resize(cropped_frame, (256, 256))
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Display the cropped frame safely
            ax.clear()
            ax.imshow(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
            plt.pause(0.05)

            # Prepare input for the model
            img = resized_rgb[np.newaxis, np.newaxis, ...]
            observation = {
                "image_primary": img,
                "timestep_pad_mask": np.array([[True]])
            }

            # Model inference
            action = model.sample_actions(
                observation,
                task,
                unnormalization_statistics=model.dataset_statistics["taco_play"]["action"],
                rng=rng_key
            )

            # Write action to a file
            print(action)
            local_action_file = "action.txt"
            with open(local_action_file, "w") as f:
                line = ",".join(f"{val:.6f}" for val in action[0][0])
                f.write(line + "\n")

            # Upload the file
            sftp.put(local_action_file, remote_path)

        except jax.errors.UnexpectedTracerError as tracer_error:
            print(f"JAX Tracer Error: {tracer_error}")
            break

        except Exception as e:
            print(f"General error encountered: {e}")
            traceback.print_exc()
            break

        # Wait 1 second before the next iteration
        time.sleep(0.25)

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    # Cleanup resources
    cap.release()
    sftp.close()
    ssh.close()
    plt.close(fig)

# import os
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# import cv2
# import time
# import numpy as np
# from PIL import Image
# import jax
# from octo.model.octo_model import OctoModel
# import paramiko
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button

# # Remote SSH details from environment variables
# host = os.getenv("SSH_HOST", "127.0.0.1")  # Default to localhost if not set
# user = os.getenv("SSH_USER", "user")
# password = os.getenv("SSH_PASSWORD", "password")
# remote_path = "/home/caleb/robochem_steps/octo_action.txt"

# # Set up SSH & SFTP connection once outside the loop for efficiency
# ssh = paramiko.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect(host, username=user, password=password)
# sftp = ssh.open_sftp()

# # Load the pretrained model once
# model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")

# # Create a fixed task (adjust the text prompt as needed)
# task_text = "reach up in the air"
# task = model.create_tasks(texts=[task_text])

# # Open the camera
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise Exception("Could not open camera.")

# # Initialize cropping parameters
# default_crop_x = 0
# default_crop_y = 0
# crop_width = 256
# crop_height = 256

# # Callback function to update task input
# def update_task_input(event):
#     global task_text, task
#     print("Button clicked. Enter new task:")
#     task_text = input("New task: ")
#     task = model.create_tasks(texts=[task_text])
#     print(f"Task updated to: {task_text}")

# try:
#     fig, ax = plt.subplots()
#     plt.subplots_adjust(bottom=0.35)

#     # Initialize sliders for cropping
#     ax_slider_x = plt.axes([0.25, 0.2, 0.65, 0.03])
#     ax_slider_y = plt.axes([0.25, 0.15, 0.65, 0.03])
#     slider_x = Slider(ax_slider_x, 'Crop X', 0, 1, valinit=0, valstep=0.01)
#     slider_y = Slider(ax_slider_y, 'Crop Y', 0, 1, valinit=0, valstep=0.01)

#     # Add a button for task input update
#     ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
#     button = Button(ax_button, 'Update Task')
#     button.on_clicked(update_task_input)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to read from camera. Exiting.")
#             break

#         # Get cropping parameters from sliders
#         crop_x = int(slider_x.val * (frame.shape[1] - crop_width))
#         crop_y = int(slider_y.val * (frame.shape[0] - crop_height))

#         # Crop the frame based on slider values
#         cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        
#         # Preprocess the frame for the model: resize and convert to RGB
#         resized = cv2.resize(cropped_frame, (256, 256))
#         resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

#         # Display the cropped frame
#         ax.clear()
#         ax.imshow(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
#         plt.draw()
#         plt.pause(0.001)

#         try:
#             # Model expects [1, 1, H, W, C]
#             img = resized_rgb[np.newaxis, np.newaxis, ...]
            
#             observation = {
#                 "image_primary": img,
#                 "timestep_pad_mask": np.array([[True]])
#             }

#             # Sample action from the model
#             action = model.sample_actions(
#                 observation,
#                 task,
#                 unnormalization_statistics=model.dataset_statistics["taco_play"]["action"],
#                 rng=jax.random.PRNGKey(0)
#             )

#             # Print the action
#             save_action = action[0]
#             # print("Action:", action[0])

#             # Write the action to a local file
#             local_action_file = "action.txt"
#             with open(local_action_file, "w") as f:
#                     # Format each value to 6 decimal places and then join with commas
#                     line = ",".join(f"{val:.6f}" for val in action[0][0])
#                     f.write(line + "\n")
#                     # f.write(str(action[0][0]))

#             # Upload the local file to the remote server
#             sftp.put(local_action_file, remote_path)
#         except Exception as e:
#             print(f"Error during model action sampling or file writing: {e}")

#         # Wait 1 second before the next iteration
#         time.sleep(0.25)

# except KeyboardInterrupt:
#     print("Interrupted by user.")
# finally:
#     cap.release()
#     sftp.close()
#     ssh.close()
#     plt.close(fig)

# import os
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# import cv2
# import time
# import numpy as np
# from PIL import Image
# import jax
# from octo.model.octo_model import OctoModel
# import paramiko
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button

# # Remote SSH details from environment variables
# host = os.getenv("SSH_HOST", "127.0.0.1")  # Default to localhost if not set
# user = os.getenv("SSH_USER", "user")
# password = os.getenv("SSH_PASSWORD", "password")
# remote_path = "/home/caleb/robochem_steps/octo_action.txt"

# # Set up SSH & SFTP connection once outside the loop for efficiency
# ssh = paramiko.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect(host, username=user, password=password)
# sftp = ssh.open_sftp()

# # Load the pretrained model once
# model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")

# # Create a fixed task (adjust the text prompt as needed)
# task_text = "reach up in the air"
# task = model.create_tasks(texts=[task_text])

# # Open the camera
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise Exception("Could not open camera.")

# # Initialize cropping parameters
# crop_width = 256
# crop_height = 256

# # Callback function to update task input
# def update_task_input(event):
#     global task_text, task
#     print("Button clicked. Enter new task:")
#     task_text = input("New task: ")
#     task = model.create_tasks(texts=[task_text])
#     print(f"Task updated to: {task_text}")

# try:
#     fig, ax = plt.subplots()
#     plt.subplots_adjust(bottom=0.35)

#     # Initialize sliders for cropping
#     ax_slider_x = plt.axes([0.25, 0.2, 0.65, 0.03])
#     ax_slider_y = plt.axes([0.25, 0.15, 0.65, 0.03])
#     slider_x = Slider(ax_slider_x, 'Crop X', 0, 1, valinit=0, valstep=0.01)
#     slider_y = Slider(ax_slider_y, 'Crop Y', 0, 1, valinit=0, valstep=0.01)

#     # Add a button for task input update
#     ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
#     button = Button(ax_button, 'Update Task')
#     button.on_clicked(update_task_input)

#     rng_key = jax.random.PRNGKey(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to read from camera. Exiting.")
#             break

#         # Get cropping parameters from sliders
#         crop_x = int(slider_x.val * (frame.shape[1] - crop_width))
#         crop_y = int(slider_y.val * (frame.shape[0] - crop_height))

#         # Crop the frame based on slider values
#         cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

#         # Preprocess the frame for the model: resize and convert to RGB
#         resized = cv2.resize(cropped_frame, (256, 256))
#         resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

#         # Display the cropped frame safely
#         ax.imshow(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
#         plt.pause(0.1)

#         try:
#             # Model expects [1, 1, H, W, C]
#             img = resized_rgb[np.newaxis, np.newaxis, ...]

#             observation = {
#                 "image_primary": img,
#                 "timestep_pad_mask": np.array([[True]])
#             }

#             # Sample action from the model
#             action = model.sample_actions(
#                 observation,
#                 task,
#                 unnormalization_statistics=model.dataset_statistics["taco_play"]["action"],
#                 rng=rng_key
#             )

#             # Print the action
#             save_action = action[0]

#             # Write the action to a local file
#             local_action_file = "action.txt"
#             with open(local_action_file, "w") as f:
#                 line = ",".join(f"{val:.6f}" for val in action[0][0])
#                 f.write(line + "\n")

#             # Upload the local file to the remote server
#             sftp.put(local_action_file, remote_path)
#         except Exception as e:
#             print(f"Error during model action sampling or file writing: {e}")

#         time.sleep(0.1)

# except KeyboardInterrupt:
#     print("Interrupted by user.")
# finally:
#     cap.release()
#     sftp.close()
#     ssh.close()
#     plt.close(fig)


# import os
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# import cv2
# import time
# import numpy as np
# from PIL import Image
# import jax
# from octo.model.octo_model import OctoModel
# import paramiko
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button

# # Remote SSH details from environment variables
# host = os.getenv("SSH_HOST", "127.0.0.1")  # Default to localhost if not set
# user = os.getenv("SSH_USER", "user")
# password = os.getenv("SSH_PASSWORD", "password")
# remote_path = "/home/caleb/robochem_steps/octo_action.txt"

# # Set up SSH & SFTP connection once outside the loop for efficiency
# ssh = paramiko.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect(host, username=user, password=password)
# sftp = ssh.open_sftp()

# # Load the pretrained model once
# model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")

# # Create a fixed task (adjust the text prompt as needed)
# task_text = "reach up in the air"
# task = model.create_tasks(texts=[task_text])

# # Open the camera
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise Exception("Could not open camera.")

# # Initialize cropping parameters
# default_crop_x = 0
# default_crop_y = 0
# crop_width = 256
# crop_height = 256

# # Callback function to update task input
# def update_task_input(event):
#     global task_text, task
#     print("Button clicked. Enter new task:")
#     task_text = input("New task: ")
#     task = model.create_tasks(texts=[task_text])
#     print(f"Task updated to: {task_text}")

# try:
#     fig, ax = plt.subplots()
#     plt.subplots_adjust(bottom=0.35)

#     # Initialize sliders for cropping
#     ax_slider_x = plt.axes([0.25, 0.2, 0.65, 0.03])
#     ax_slider_y = plt.axes([0.25, 0.15, 0.65, 0.03])
#     slider_x = Slider(ax_slider_x, 'Crop X', 0, 1, valinit=0, valstep=0.01)
#     slider_y = Slider(ax_slider_y, 'Crop Y', 0, 1, valinit=0, valstep=0.01)

#     # Add a button for task input update
#     ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
#     button = Button(ax_button, 'Update Task')
#     button.on_clicked(update_task_input)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to read from camera. Exiting.")
#             break

#         # Get cropping parameters from sliders
#         crop_x = int(slider_x.val * (frame.shape[1] - crop_width))
#         crop_y = int(slider_y.val * (frame.shape[0] - crop_height))

#         # Crop the frame based on slider values
#         cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        
#         # Preprocess the frame for the model: resize and convert to RGB
#         resized = cv2.resize(cropped_frame, (256, 256))
#         resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

#         # Display the cropped frame
#         ax.clear()
#         ax.imshow(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
#         plt.draw()
#         plt.pause(0.001)

#         try:
#             # Model expects [1, 1, H, W, C]
#             img = resized_rgb[np.newaxis, np.newaxis, ...]
            
#             observation = {
#                 "image_primary": img,
#                 "timestep_pad_mask": np.array([[True]])
#             }

#             # Sample action from the model
#             action = model.sample_actions(
#                 observation,
#                 task,
#                 unnormalization_statistics=model.dataset_statistics["taco_play"]["action"],
#                 rng=jax.random.PRNGKey(0)
#             )

#             # Print the action
#             save_action = action[0]
#             # print("Action:", action[0])

#             # Write the action to a local file
#             local_action_file = "action.txt"
#             with open(local_action_file, "w") as f:
#                     # Format each value to 6 decimal places and then join with commas
#                     line = ",".join(f"{val:.6f}" for val in action[0][0])
#                     f.write(line + "\n")
#                     # f.write(str(action[0][0]))

#             # Upload the local file to the remote server
#             sftp.put(local_action_file, remote_path)
#         except Exception as e:
#             print(f"Error during model action sampling or file writing: {e}")

#         # Wait 1 second before the next iteration
#         time.sleep(0.25)

# except KeyboardInterrupt:
#     print("Interrupted by user.")
# finally:
#     cap.release()
#     sftp.close()
#     ssh.close()
#     plt.close(fig)


# import os
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# import cv2
# import time
# import numpy as np
# from PIL import Image
# import jax
# from octo.model.octo_model import OctoModel
# import paramiko
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, TextBox

# # Remote SSH details from environment variables
# host = os.getenv("SSH_HOST", "127.0.0.1")  # Default to localhost if not set
# user = os.getenv("SSH_USER", "user")
# password = os.getenv("SSH_PASSWORD", "password")
# remote_path = "/home/caleb/robochem_steps/octo_action.txt"

# # Set up SSH & SFTP connection once outside the loop for efficiency
# ssh = paramiko.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect(host, username=user, password=password)
# sftp = ssh.open_sftp()

# # Load the pretrained model once
# model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")

# # Create a fixed task (adjust the text prompt as needed)
# task_text = "reach up in the air"
# task = model.create_tasks(texts=[task_text])

# # Open the camera
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise Exception("Could not open camera.")

# # Initialize cropping parameters
# default_crop_x = 0
# default_crop_y = 0
# crop_width = 256
# crop_height = 256

# # Callback function to update task input
# def update_task_input(text):
#     global task_text, task
#     task_text = text
#     task = model.create_tasks(texts=[task_text])
#     print(f"Task updated to: {task_text}")

# try:
#     fig, ax = plt.subplots()
#     plt.subplots_adjust(bottom=0.35)

#     # Initialize sliders for cropping
#     ax_slider_x = plt.axes([0.25, 0.2, 0.65, 0.03])
#     ax_slider_y = plt.axes([0.25, 0.15, 0.65, 0.03])
#     slider_x = Slider(ax_slider_x, 'Crop X', 0, 1, valinit=0, valstep=0.01)
#     slider_y = Slider(ax_slider_y, 'Crop Y', 0, 1, valinit=0, valstep=0.01)

#     # Add a text box for task input
#     ax_textbox = plt.axes([0.25, 0.05, 0.65, 0.05])
#     textbox = TextBox(ax_textbox, 'Task Input', initial=task_text)
#     textbox.on_submit(update_task_input)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to read from camera. Exiting.")
#             break

#         # Get cropping parameters from sliders
#         crop_x = int(slider_x.val * (frame.shape[1] - crop_width))
#         crop_y = int(slider_y.val * (frame.shape[0] - crop_height))

#         # Crop the frame based on slider values
#         cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        
#         # Preprocess the frame for the model: resize and convert to RGB
#         resized = cv2.resize(cropped_frame, (256, 256))
#         resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

#         # Display the cropped frame
#         ax.clear()
#         ax.imshow(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
#         plt.draw()
#         plt.pause(0.001)

#         # Model expects [1, 1, H, W, C]
#         img = resized_rgb[np.newaxis, np.newaxis, ...]
        
#         observation = {
#             "image_primary": img,
#             "timestep_pad_mask": np.array([[True]])
#         }

#         # Sample action from the model
#         action = model.sample_actions(
#             observation,
#             task,
#             unnormalization_statistics=model.dataset_statistics["taco_play"]["action"],  # Ensure this key exists and is structured correctly
#             rng=jax.random.PRNGKey(0)
#         )

#         # Print the action
#         save_action = action[0]
#         # print("Action:", action[0])

#         # Write the action to a local file
#         local_action_file = "action.txt"
#         with open(local_action_file, "w") as f:
#                 # Format each value to 6 decimal places and then join with commas
#                 line = ",".join(f"{val:.6f}" for val in action[0][0])
#                 f.write(line + "\n")
#                 # f.write(str(action[0][0]))

#         # Upload the local file to the remote server
#         sftp.put(local_action_file, remote_path)

#         # Wait 1 second before the next iteration
#         time.sleep(0.25)

# except KeyboardInterrupt:
#     print("Interrupted by user.")
# finally:
#     cap.release()
#     sftp.close()
#     ssh.close()
#     plt.close(fig)

# import os
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# import cv2
# import time
# import numpy as np
# from PIL import Image
# import jax
# from octo.model.octo_model import OctoModel
# import paramiko
# from pynput import keyboard
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider

# # Remote SSH details from environment variables
# host = os.getenv("SSH_HOST", "127.0.0.1")  # Default to localhost if not set
# user = os.getenv("SSH_USER", "user")
# password = os.getenv("SSH_PASSWORD", "password")
# remote_path = "/home/caleb/robochem_steps/octo_action.txt"

# # Set up SSH & SFTP connection once outside the loop for efficiency
# ssh = paramiko.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect(host, username=user, password=password)
# sftp = ssh.open_sftp()

# # Load the pretrained model once
# model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")

# # Create a fixed task (adjust the text prompt as needed)
# task_text = "reach up in the air"
# task = model.create_tasks(texts=[task_text])

# # Open the camera
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise Exception("Could not open camera.")

# # Define a function to handle key presses
# def on_press(key):
#     global task_text, task
#     try:
#         if key.char == 'g':
#             print("\n'g' key detected. Enter a new task input:")
#             task_text = input("New task: ")
#             task = model.create_tasks(texts=[task_text])
#             print(f"Task updated to: {task_text}")
#     except AttributeError:
#         pass

# # Start the keyboard listener
# listener = keyboard.Listener(on_press=on_press)
# listener.start()

# # Initialize cropping parameters
# default_crop_x = 0
# default_crop_y = 0
# crop_width = 256
# crop_height = 256

# try:
#     fig, ax = plt.subplots()
#     plt.subplots_adjust(bottom=0.2)

#     # Initialize sliders for cropping
#     ax_slider_x = plt.axes([0.25, 0.1, 0.65, 0.03])
#     ax_slider_y = plt.axes([0.25, 0.05, 0.65, 0.03])
#     slider_x = Slider(ax_slider_x, 'Crop X', 0, 1, valinit=0, valstep=0.01)
#     slider_y = Slider(ax_slider_y, 'Crop Y', 0, 1, valinit=0, valstep=0.01)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to read from camera. Exiting.")
#             break

#         # Get cropping parameters from sliders
#         crop_x = int(slider_x.val * (frame.shape[1] - crop_width))
#         crop_y = int(slider_y.val * (frame.shape[0] - crop_height))

#         # Crop the frame based on slider values
#         cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        
#         # Preprocess the frame for the model: resize and convert to RGB
#         resized = cv2.resize(cropped_frame, (256, 256))
#         resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

#         # Display the cropped frame
#         ax.clear()
#         ax.imshow(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
#         plt.draw()
#         plt.pause(0.001)

#         # Model expects [1, 1, H, W, C]
#         img = resized_rgb[np.newaxis, np.newaxis, ...]
        
#         observation = {
#             "image_primary": img,
#             "timestep_pad_mask": np.array([[True]])
#         }

#         # Sample action from the model
#         action = model.sample_actions(
#             observation,
#             task,
#             unnormalization_statistics=model.dataset_statistics["taco_play"]["action"],  # Ensure this key exists and is structured correctly
#             rng=jax.random.PRNGKey(0)
#         )

#         # Print the action
#         save_action = action[0]
#         # print("Action:", action[0])

#         # Write the action to a local file
#         local_action_file = "action.txt"
#         with open(local_action_file, "w") as f:
#                 # Format each value to 6 decimal places and then join with commas
#                 line = ",".join(f"{val:.6f}" for val in action[0][0])
#                 f.write(line + "\n")
#                 # f.write(str(action[0][0]))

#         # Upload the local file to the remote server
#         sftp.put(local_action_file, remote_path)

#         # Wait 1 second before the next iteration
#         time.sleep(0.25)

# except KeyboardInterrupt:
#     print("Interrupted by user.")
# finally:
#     cap.release()
#     sftp.close()
#     ssh.close()
#     listener.stop()
#     plt.close(fig)

# # import os
# # os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# # import cv2
# # import time
# # import numpy as np
# # from PIL import Image
# # import jax
# # from octo.model.octo_model import OctoModel
# # import paramiko
# # from pynput import keyboard

# # # Remote SSH details
# # host = "128.138.244.100"  # e.g. "192.168.1.10"
# # user = "caleb"
# # password = "HIROlab322"
# # remote_path = "/home/caleb/robochem_steps/octo_action.txt"

# # # Set up SSH & SFTP connection once outside the loop for efficiency
# # ssh = paramiko.SSHClient()
# # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# # ssh.connect(host, username=user, password=password)
# # sftp = ssh.open_sftp()

# # # Load the pretrained model once
# # model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")

# # # Create a fixed task (adjust the text prompt as needed)
# # task_text = "reach up in the air"
# # task = model.create_tasks(texts=[task_text])

# # # Open the camera
# # cap = cv2.VideoCapture(0)
# # if not cap.isOpened():
# #     raise Exception("Could not open camera.")

# # # Define a function to handle key presses
# # def on_press(key):
# #     global task_text, task
# #     try:
# #         if key.char == 'g':
# #             print("\n'g' key detected. Enter a new task input:")
# #             task_text = input("New task: ")
# #             task = model.create_tasks(texts=[task_text])
# #             print(f"Task updated to: {task_text}")
# #     except AttributeError:
# #         pass

# # # Start the keyboard listener
# # listener = keyboard.Listener(on_press=on_press)
# # listener.start()

# # try:
# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             print("Failed to read from camera. Exiting.")
# #             break
        
# #         # Preprocess the frame for the model: resize and convert to RGB
# #         resized = cv2.resize(frame, (256, 256))
# #         resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

# #         # Model expects [1, 1, H, W, C]
# #         img = resized_rgb[np.newaxis, np.newaxis, ...]
        
# #         observation = {
# #             "image_primary": img,
# #             "timestep_pad_mask": np.array([[True]])
# #         }

# #         # Sample action from the model
# #         action = model.sample_actions(
# #             observation,
# #             task,
# #             unnormalization_statistics=model.dataset_statistics["taco_play"]["action"],  # Ensure this key exists and is structured correctly
# #             rng=jax.random.PRNGKey(0)
# #         )

# #         # Print the action
# #         save_action = action[0]
# #         # print("Action:", action[0])

# #         # Write the action to a local file
# #         local_action_file = "action.txt"
# #         with open(local_action_file, "w") as f:
# #                 # Format each value to 6 decimal places and then join with commas
# #                 line = ",".join(f"{val:.6f}" for val in action[0][0])
# #                 f.write(line + "\n")
# #                 # f.write(str(action[0][0]))

# #         # Upload the local file to the remote server
# #         sftp.put(local_action_file, remote_path)

# #         # Wait 1 second before the next iteration
# #         time.sleep(0.25)

# # except KeyboardInterrupt:
# #     print("Interrupted by user.")
# # finally:
# #     cap.release()
# #     sftp.close()
# #     ssh.close()
# #     listener.stop()