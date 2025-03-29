import os
import cv2
import time
import rclpy
import threading
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool  # <-- Added

SAVE_ROOT = "/home/caleb/datasets/octo_test"

class JointStateListener(Node):
    def __init__(self):
        super().__init__('joint_state_listener')
        self.joint_state = None
        self.twist = None
        self.gripper_open = None
        self.lock = threading.Lock()

        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.create_subscription(TwistStamped, '/robot_twist', self.twist_callback, 10)
        self.create_subscription(Bool, '/gripper_open', self.gripper_callback, 10)  # <-- New

    def joint_callback(self, msg):
        with self.lock:
            self.joint_state = {
                'position': np.array(msg.position, dtype=np.float32),
                'velocity': np.array(msg.velocity, dtype=np.float32),
                'effort': np.array(msg.effort, dtype=np.float32)
            }

    def twist_callback(self, msg):
        with self.lock:
            self.twist = np.array([
                msg.twist.linear.x,
                msg.twist.linear.y,
                msg.twist.linear.z,
                msg.twist.angular.x,
                msg.twist.angular.y,
                msg.twist.angular.z
            ], dtype=np.float32)

    def gripper_callback(self, msg):
        with self.lock:
            self.gripper_open = 1 if msg.data else 0  # Convert to 1=open, 0=closed

    def get_latest(self):
        with self.lock:
            if self.joint_state is None or self.twist is None or self.gripper_open is None:
                return None
            return {
                **self.joint_state,
                'twist': self.twist.copy(),
                'gripper_open': self.gripper_open
            }

class CameraFeedApp:
    def __init__(self, wrist_cam_id, third_person_ids):
        self.root = Tk()
        self.root.title("Octo Data Collector")
        self.root.geometry("1300x900")

        self.lang_var = StringVar()
        self.recording = False
        os.makedirs(SAVE_ROOT, exist_ok=True)
        existing = [d for d in os.listdir(SAVE_ROOT) if d.startswith("task_")]
        ids = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
        self.task_id = max(ids) if ids else 0
        self.step_id = 0

        self.wrist_cam_id = wrist_cam_id
        self.third_person_ids = third_person_ids
        self.caps = {
            "wrist": cv2.VideoCapture(wrist_cam_id),
            "third_person_1": cv2.VideoCapture(third_person_ids[0]),
            "third_person_2": cv2.VideoCapture(third_person_ids[1])
        }

        rclpy.init(args=None)
        self.ros_node = JointStateListener()

        Label(self.root, text="Language Instruction:").pack()
        Entry(self.root, textvariable=self.lang_var, width=60).pack()

        Button(self.root, text="Start Recording", command=self.start_recording).pack(pady=5)
        Button(self.root, text="Stop Recording", command=self.stop_recording).pack(pady=5)

        self.frame = Frame(self.root)
        self.frame.pack()

        self.labels = {}
        for i, key in enumerate(self.caps.keys()):
            label = Label(self.frame)
            label.grid(row=0, column=i)
            self.labels[key] = label

        self.running = True
        self.update_thread = threading.Thread(target=self.update_loop)
        self.update_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def update_loop(self):
        ros_spin_thread = threading.Thread(target=rclpy.spin, args=(self.ros_node,), daemon=True)
        ros_spin_thread.start()

        while self.running:
            frames = {}
            for key, cap in self.caps.items():
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (128, 128)) if key == "wrist" else cv2.resize(frame, (256, 256))

                frames[key] = frame

                img = ImageTk.PhotoImage(Image.fromarray(frame))
                self.labels[key].config(image=img)
                self.labels[key].image = img

            if self.recording:
                self.save_step(frames)

            time.sleep(0.1)

    def save_step(self, frames):
        state = self.ros_node.get_latest()
        if state is None:
            print("⚠️ Skipping save — missing joint, twist, or gripper state.")
            return

        task_dir = os.path.join(SAVE_ROOT, f"task_{self.task_id:05d}")
        os.makedirs(task_dir, exist_ok=True)

        save_path = os.path.join(task_dir, f"step_{self.step_id:05d}.npz")
        np.savez_compressed(
            save_path,
            timestamp=time.time(),
            language_instruction=self.lang_var.get(),
            rgb_wrist=frames["wrist"],
            rgb_third_person_1=frames["third_person_1"],
            rgb_third_person_2=frames["third_person_2"],
            joint_positions=state["position"],
            joint_velocities=state["velocity"],
            joint_efforts=state["effort"],
            ee_twist=state["twist"],
            gripper_open=state["gripper_open"]
        )

        print(f"💾 Saved {save_path} | Gripper: {'Open' if state['gripper_open'] else 'Closed'}")
        self.step_id += 1

    def start_recording(self):
        print("🚀 Recording started with instruction:", self.lang_var.get())
        self.task_id += 1
        self.step_id = 0
        self.recording = True

    def stop_recording(self):
        print("🛑 Recording stopped.")
        self.recording = False

    def on_close(self):
        print("👋 Closing...")
        self.running = False
        for cap in self.caps.values():
            cap.release()
        rclpy.shutdown()
        self.root.destroy()

if __name__ == "__main__":
    CameraFeedApp(wrist_cam_id=4, third_person_ids=[0, 2])

# import os
# import cv2
# import time
# import rclpy
# import threading
# import numpy as np
# from tkinter import *
# from PIL import Image, ImageTk
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from geometry_msgs.msg import TwistStamped

# SAVE_ROOT = "/home/caleb/datasets/octo_test"

# class JointStateListener(Node):
#     def __init__(self):
#         super().__init__('joint_state_listener')
#         self.joint_state = None
#         self.twist = None
#         self.lock = threading.Lock()

#         self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
#         self.create_subscription(TwistStamped, '/robot_twist', self.twist_callback, 10)

#     def joint_callback(self, msg):
#         with self.lock:
#             self.joint_state = {
#                 'position': np.array(msg.position, dtype=np.float32),
#                 'velocity': np.array(msg.velocity, dtype=np.float32),
#                 'effort': np.array(msg.effort, dtype=np.float32)
#             }

#     def twist_callback(self, msg):
#         with self.lock:
#             self.twist = np.array([
#                 msg.twist.linear.x,
#                 msg.twist.linear.y,
#                 msg.twist.linear.z,
#                 msg.twist.angular.x,
#                 msg.twist.angular.y,
#                 msg.twist.angular.z
#             ], dtype=np.float32)

#     def get_latest(self):
#         with self.lock:
#             if self.joint_state is None or self.twist is None:
#                 return None
#             return {**self.joint_state, 'twist': self.twist.copy()}

# class CameraFeedApp:
#     def __init__(self, wrist_cam_id, third_person_ids):
#         self.root = Tk()
#         self.root.title("Octo Data Collector")
#         self.root.geometry("1300x900")

#         self.lang_var = StringVar()
#         self.recording = False
#         os.makedirs(SAVE_ROOT, exist_ok=True)
#         existing = [d for d in os.listdir(SAVE_ROOT) if d.startswith("task_")]
#         ids = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
#         self.task_id = max(ids) if ids else 0
#         self.step_id = 0

#         self.wrist_cam_id = wrist_cam_id
#         self.third_person_ids = third_person_ids
#         self.caps = {
#             "wrist": cv2.VideoCapture(wrist_cam_id),
#             "third_person_1": cv2.VideoCapture(third_person_ids[0]),
#             "third_person_2": cv2.VideoCapture(third_person_ids[1])
#         }

#         rclpy.init(args=None)
#         self.ros_node = JointStateListener()

#         Label(self.root, text="Language Instruction:").pack()
#         Entry(self.root, textvariable=self.lang_var, width=60).pack()

#         Button(self.root, text="Start Recording", command=self.start_recording).pack(pady=5)
#         Button(self.root, text="Stop Recording", command=self.stop_recording).pack(pady=5)

#         self.frame = Frame(self.root)
#         self.frame.pack()

#         self.labels = {}
#         for i, key in enumerate(self.caps.keys()):
#             label = Label(self.frame)
#             label.grid(row=0, column=i)
#             self.labels[key] = label

#         self.running = True
#         self.update_thread = threading.Thread(target=self.update_loop)
#         self.update_thread.start()

#         self.root.protocol("WM_DELETE_WINDOW", self.on_close)
#         self.root.mainloop()

#     def update_loop(self):
#         ros_spin_thread = threading.Thread(target=rclpy.spin, args=(self.ros_node,), daemon=True)
#         ros_spin_thread.start()

#         while self.running:
#             frames = {}
#             for key, cap in self.caps.items():
#                 ret, frame = cap.read()
#                 if not ret:
#                     frame = np.zeros((480, 640, 3), dtype=np.uint8)
#                 else:
#                     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     frame = cv2.resize(frame, (128, 128)) if key == "wrist" else cv2.resize(frame, (256, 256))

#                 frames[key] = frame

#                 img = ImageTk.PhotoImage(Image.fromarray(frame))
#                 self.labels[key].config(image=img)
#                 self.labels[key].image = img

#             if self.recording:
#                 self.save_step(frames)

#             time.sleep(0.1)

#     def save_step(self, frames):
#         state = self.ros_node.get_latest()
#         if state is None:
#             print("⚠️ Skipping save — joint or twist state not yet received.")
#             return

#         task_dir = os.path.join(SAVE_ROOT, f"task_{self.task_id:05d}")
#         os.makedirs(task_dir, exist_ok=True)

#         save_path = os.path.join(task_dir, f"step_{self.step_id:05d}.npz")
#         np.savez_compressed(
#             save_path,
#             timestamp=time.time(),
#             language_instruction=self.lang_var.get(),
#             rgb_wrist=frames["wrist"],
#             rgb_third_person_1=frames["third_person_1"],
#             rgb_third_person_2=frames["third_person_2"],
#             joint_positions=state["position"],
#             joint_velocities=state["velocity"],
#             joint_efforts=state["effort"],
#             ee_twist=state["twist"]
#         )

#         print(f"💾 Saved {save_path}")
#         self.step_id += 1

#     def start_recording(self):
#         print("🚀 Recording started with instruction:", self.lang_var.get())
#         self.task_id += 1
#         self.step_id = 0
#         self.recording = True

#     def stop_recording(self):
#         print("🛑 Recording stopped.")
#         self.recording = False

#     def on_close(self):
#         print("👋 Closing...")
#         self.running = False
#         for cap in self.caps.values():
#             cap.release()
#         rclpy.shutdown()
#         self.root.destroy()

# if __name__ == "__main__":
#     CameraFeedApp(wrist_cam_id=4, third_person_ids=[0, 2])
