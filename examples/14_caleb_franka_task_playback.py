import os
import numpy as np
import cv2
import time
import threading
from tkinter import *
from PIL import Image, ImageTk

class TaskPlaybackApp:
    def __init__(self, root_dir):
        self.root = Tk()
        self.root.title("Octo Task Playback")
        self.root.geometry("1400x1000")

        self.root_dir = root_dir
        self.task_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.task_var = StringVar()
        self.task_menu = OptionMenu(self.root, self.task_var, *self.task_dirs, command=self.load_task)
        self.task_menu.pack(pady=10)

        self.lang_label_text = StringVar()
        self.lang_label = Label(self.root, textvariable=self.lang_label_text, font=("Arial", 14), height=2)
        self.lang_label.pack(pady=5, fill=X)

        self.joint_state_text = StringVar()
        self.joint_state_label = Label(self.root, textvariable=self.joint_state_text, font=("Courier New", 10), height=6, justify=LEFT)
        self.joint_state_label.pack(pady=5, fill=X)

        self.frame = Frame(self.root)
        self.frame.pack()

        self.image_labels = {
            "wrist": Label(self.frame),
            "third_person_1": Label(self.frame),
            "third_person_2": Label(self.frame)
        }
        self.image_labels["wrist"].grid(row=0, column=0)
        self.image_labels["third_person_1"].grid(row=0, column=1)
        self.image_labels["third_person_2"].grid(row=0, column=2)

        self.step_files = []
        self.step_index = 0
        self.playback_thread = None
        self.playing = False

        self.root.mainloop()

    def load_task(self, task_name):
        task_dir = os.path.join(self.root_dir, task_name)
        self.step_files = sorted([
            os.path.join(task_dir, f) for f in os.listdir(task_dir) if f.endswith('.npz')
        ])
        self.step_index = 0
        self.playing = True
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.playback_thread = threading.Thread(target=self.auto_play, daemon=True)
            self.playback_thread.start()

    def auto_play(self):
        while self.playing and self.step_index < len(self.step_files):
            self.display_step()
            self.step_index += 1
            time.sleep(0.1)

    def load_step(self, path):
        data = np.load(path, allow_pickle=True)
        return {
            "lang": str(data["language_instruction"]),
            "joint": {
                "position": data["joint_positions"],
                "velocity": data["joint_velocities"],
                "effort": data["joint_efforts"]
            },
            "twist": data["ee_twist"],
            "gripper_open": int(data["gripper_open"]),  # <-- Added line
            "images": {
                "wrist": data["rgb_wrist"],
                "third_person_1": data["rgb_third_person_1"],
                "third_person_2": data["rgb_third_person_2"]
            }
        }

    def display_step(self):
        if not self.step_files:
            return

        step_path = self.step_files[self.step_index]
        step_data = self.load_step(step_path)

        self.lang_label_text.set(f"Language Instruction: {step_data['lang']}")

        pos = np.round(step_data['joint']['position'], 2)
        vel = np.round(step_data['joint']['velocity'], 2)
        eff = np.round(step_data['joint']['effort'], 2)
        twist = np.round(step_data['twist'], 4)
        gripper_str = "Open" if step_data["gripper_open"] else "Closed"  # <-- Added line

        joint_text = (
            f"Position: {pos}\n"
            f"Velocity: {vel}\n"
            f"Effort: {eff}\n"
            f"EE Twist: [vx: {twist[0]}, vy: {twist[1]}, vz: {twist[2]}, wx: {twist[3]}, wy: {twist[4]}, wz: {twist[5]}]\n"
            f"Gripper: {gripper_str}"  # <-- Added line
        )
        self.joint_state_text.set(joint_text)

        for key, img in step_data['images'].items():
            img = Image.fromarray(img.astype(np.uint8))
            imgtk = ImageTk.PhotoImage(image=img)
            self.image_labels[key].config(image=imgtk)
            self.image_labels[key].image = imgtk


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python task_playback.py /path/to/dataset_dir")
        exit(1)

    root_dir = sys.argv[1]
    TaskPlaybackApp(root_dir)


# import os
# import numpy as np
# import cv2
# import time
# import threading
# from tkinter import *
# from PIL import Image, ImageTk

# class TaskPlaybackApp:
#     def __init__(self, root_dir):
#         self.root = Tk()
#         self.root.title("Octo Task Playback")
#         self.root.geometry("1400x1000")

#         self.root_dir = root_dir
#         self.task_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
#         self.task_var = StringVar()
#         self.task_menu = OptionMenu(self.root, self.task_var, *self.task_dirs, command=self.load_task)
#         self.task_menu.pack(pady=10)

#         self.lang_label_text = StringVar()
#         self.lang_label = Label(self.root, textvariable=self.lang_label_text, font=("Arial", 14), height=2)
#         self.lang_label.pack(pady=5, fill=X)

#         self.joint_state_text = StringVar()
#         self.joint_state_label = Label(self.root, textvariable=self.joint_state_text, font=("Courier New", 10), height=6, justify=LEFT)
#         self.joint_state_label.pack(pady=5, fill=X)

#         self.frame = Frame(self.root)
#         self.frame.pack()

#         self.image_labels = {
#             "wrist": Label(self.frame),
#             "third_person_1": Label(self.frame),
#             "third_person_2": Label(self.frame)
#         }
#         self.image_labels["wrist"].grid(row=0, column=0)
#         self.image_labels["third_person_1"].grid(row=0, column=1)
#         self.image_labels["third_person_2"].grid(row=0, column=2)

#         self.step_files = []
#         self.step_index = 0
#         self.playback_thread = None
#         self.playing = False

#         self.root.mainloop()

#     def load_task(self, task_name):
#         task_dir = os.path.join(self.root_dir, task_name)
#         self.step_files = sorted([
#             os.path.join(task_dir, f) for f in os.listdir(task_dir) if f.endswith('.npz')
#         ])
#         self.step_index = 0
#         self.playing = True
#         if self.playback_thread is None or not self.playback_thread.is_alive():
#             self.playback_thread = threading.Thread(target=self.auto_play, daemon=True)
#             self.playback_thread.start()

#     def auto_play(self):
#         while self.playing and self.step_index < len(self.step_files):
#             self.display_step()
#             self.step_index += 1
#             time.sleep(0.1)

#     def load_step(self, path):
#         data = np.load(path, allow_pickle=True)
#         return {
#             "lang": str(data["language_instruction"]),
#             "joint": {
#                 "position": data["joint_positions"],
#                 "velocity": data["joint_velocities"],
#                 "effort": data["joint_efforts"]
#             },
#             "twist": data["ee_twist"],
#             "images": {
#                 "wrist": data["rgb_wrist"],
#                 "third_person_1": data["rgb_third_person_1"],
#                 "third_person_2": data["rgb_third_person_2"]
#             }
#         }

#     def display_step(self):
#         if not self.step_files:
#             return

#         step_path = self.step_files[self.step_index]
#         step_data = self.load_step(step_path)

#         self.lang_label_text.set(f"Language Instruction: {step_data['lang']}")

#         pos = np.round(step_data['joint']['position'], 2)
#         vel = np.round(step_data['joint']['velocity'], 2)
#         eff = np.round(step_data['joint']['effort'], 2)
#         twist = np.round(step_data['twist'], 4)

#         joint_text = (
#             f"Position: {pos}\n"
#             f"Velocity: {vel}\n"
#             f"Effort: {eff}\n"
#             f"EE Twist: [vx: {twist[0]}, vy: {twist[1]}, vz: {twist[2]}, wx: {twist[3]}, wy: {twist[4]}, wz: {twist[5]}]"
#         )
#         self.joint_state_text.set(joint_text)

#         for key, img in step_data['images'].items():
#             img = Image.fromarray(img.astype(np.uint8))
#             imgtk = ImageTk.PhotoImage(image=img)
#             self.image_labels[key].config(image=imgtk)
#             self.image_labels[key].image = imgtk


# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 2:
#         print("Usage: python task_playback.py /path/to/dataset_dir")
#         exit(1)

#     root_dir = sys.argv[1]
#     TaskPlaybackApp(root_dir)
