#!/bin/bash

# Find the PID of the script
process_info=$(ps aux | grep "python3 11_caleb_py_video_franka.py" | grep -v "grep")

# Check if the process is running
if [ -z "$process_info" ]; then
  echo "No process found running the script python3 11_caleb_py_video_franka.py"
  exit 1
fi

# Extract the PID
process_pid=$(echo "$process_info" | awk '{print $2}')

# Display process information
echo "Process found:"
echo "$process_info"
echo "Attempting to kill process with PID: $process_pid"

# Kill the process
sudo kill -9 "$process_pid"

# Verify if the process was successfully killed
if [ $? -eq 0 ]; then
  echo "Process $process_pid killed successfully."
else
  echo "Failed to kill process $process_pid."
fi

