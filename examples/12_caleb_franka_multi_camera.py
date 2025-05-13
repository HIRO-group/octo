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
from matplotlib.widgets import Button
import traceback

# Remote SSH details
host = os.getenv("SSH_HOST", "127.0.0.1")
user = os.getenv("SSH_USER", "user")
password = os.getenv("SSH_PASSWORD", "password")
remote_path = "/home/caleb/robochem_steps/octo_action.txt"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, username=user, password=password)
sftp = ssh.open_sftp()

# Load model
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")
# print("dataset statistics keys: ", model.dataset_statistics.keys())


# Default task
task_text = "Grab the red cylinder from the top"
task = model.create_tasks(texts=[task_text])

# Cameras
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    raise Exception("Could not open 3rd person camera.")
cap_w = cv2.VideoCapture(4)
if not cap_w.isOpened():
    raise Exception("Could not open wrist camera.")

# Task update callback
def update_task_input(event):
    global task_text, task
    print("Button clicked. Enter new task:")
    task_text = input("New task: ")
    task = model.create_tasks(texts=[task_text])
    print(f"Task updated to: {task_text}")

try:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.subplots_adjust(bottom=0.25)
    fig.suptitle(f"Task: {task_text}", fontsize=14)

    ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
    button = Button(ax_button, 'Update Task')
    button.on_clicked(update_task_input)

    rng_key = jax.random.PRNGKey(0)

    while True:
        try:
            ret, frame = cap.read()
            ret_w, frame_w = cap_w.read()
            if not ret or not ret_w:
                print("Failed to read from cameras.")
                break

            # Resize and convert to RGB
            resized = cv2.resize(frame, (256, 256))
            resized_w = cv2.resize(frame_w, (128, 128))
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            resized_w_rgb = cv2.cvtColor(resized_w, cv2.COLOR_BGR2RGB)

            # Update figure
            ax1.clear()
            ax2.clear()
            ax1.imshow(resized_rgb)
            ax1.set_title("3rd Person View")
            ax1.axis("off")
            ax2.imshow(resized_w_rgb)
            ax2.set_title("Wrist View")
            ax2.axis("off")
            fig.suptitle(f"Task: {task_text}", fontsize=14)
            plt.pause(0.05)

            # Model input
            img = resized_rgb[np.newaxis, np.newaxis, ...]
            img_w = resized_w_rgb[np.newaxis, np.newaxis, ...]
            observation = {
                # "image_primary": img,
                "image_wrist": img_w,
                "timestep_pad_mask": np.array([[True]])
            }

            # Inference
            action = model.sample_actions(
                observation,
                task,
                unnormalization_statistics=model.dataset_statistics["berkeley_fanuc_manipulation"]["action"],
                # unnormalization_statistics=model.dataset_statistics["taco_play"]["action"],
                # unnormalization_statistics=model.dataset_statistics["nyu_franka_play_dataset_converted_externally_to_rlds"]["action"], 
                rng=rng_key
            )
            print("berkeley_fanuc_manipulation statistics: ", model.dataset_statistics["berkeley_fanuc_manipulation"])
            print("All of action: ", action)
            # print("predicted action: ", action[0][0])
            # exit() 

            # Write + upload action
            with open("action.txt", "w") as f:
                line = ",".join(f"{val:.6f}" for val in action[0][0])
                f.write(line + "\n")
            sftp.put("action.txt", remote_path)

        except jax.errors.UnexpectedTracerError as tracer_error:
            print(f"JAX Tracer Error: {tracer_error}")
            break
        except Exception as e:
            print(f"General error encountered: {e}")
            traceback.print_exc()
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    cap.release()
    cap_w.release()
    sftp.close()
    ssh.close()
    plt.close(fig)
