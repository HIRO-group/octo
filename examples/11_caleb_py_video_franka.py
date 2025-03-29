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
print("dataset statistics keys: ", model.dataset_statistics.keys)
exit()

# Create a fixed task (adjust the text prompt as needed)
task_text = "grab the red tape"
task = model.create_tasks(texts=[task_text])

# Open the camera
cap = cv2.VideoCapture(6)
if not cap.isOpened():
    raise Exception("Could not open camera 3rd person.")

cap_w = cv2.VideoCapture(8)
if not cap_w.isOpened():
    raise Exception("Could not open camera wrist.")

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
    slider_x = Slider(ax_slider_x, 'Crop X', 0, 1, valinit=0.58, valstep=0.01)
    slider_y = Slider(ax_slider_y, 'Crop Y', 0, 1, valinit=0.25, valstep=0.01)

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
        except jax.errors.UnexpectedTracerError as tracer_error:
            print(f"JAX Tracer Error: {tracer_error}")
            break

        except Exception as e:
            print(f"General error encountered: {e}")
            traceback.print_exc()
            break
        try:
            ret_w, frame_w = cap_w.read()
            if not ret:
                print("Failed to read from camera wrist. Exiting.")
                break

            # Get cropping parameters from sliders
            crop_x = int(slider_x.val * (frame.shape[1] - crop_width))
            crop_y = int(slider_y.val * (frame.shape[0] - crop_height))

            # Crop the frame safely
            # cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
            cropped_frame = frame
            # Resize and convert to RGB
            resized = cv2.resize(cropped_frame, (256, 256))
            resized_w = cv2.resize(frame_w, (128, 128))
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            resized_w_rgb = cv2.cvtColor(resized_w, cv2.COLOR_BGR2RGB)

            # Display the cropped frame safely
            ax.clear()
            ax.imshow(cv2.cvtColor(resized_w, cv2.COLOR_BGR2RGB))
            plt.pause(0.05)

            # Prepare input for the model
            img = resized_rgb[np.newaxis, np.newaxis, ...]
            img_w = resized_w_rgb[np.newaxis, np.newaxis, ...]
            observation = {
                "image_primary": img,
                "image_wrist": img_w,
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
            # print(action)
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