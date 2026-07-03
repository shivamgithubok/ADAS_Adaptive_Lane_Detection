import os
import sys
import torch
import cv2
import platform
import subprocess

os.makedirs("logs", exist_ok=True)
os.makedirs("diagnostics", exist_ok=True)

with open("logs/system_report.txt", "w") as f:
    f.write("=== /etc/nv_tegra_release ===\n")
    try:
        with open("/etc/nv_tegra_release", "r") as nv_file:
            f.write(nv_file.read() + "\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
        
    f.write("=== dpkg -l | grep nvidia-l4t ===\n")
    try:
        res = subprocess.run("dpkg -l | grep nvidia-l4t", shell=True, capture_output=True, text=True)
        f.write(res.stdout + "\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
        
    f.write("=== dpkg -l | grep cuda ===\n")
    try:
        res = subprocess.run("dpkg -l | grep cuda", shell=True, capture_output=True, text=True)
        f.write(res.stdout + "\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
        
    f.write("=== Python Platform ===\n")
    f.write(f"{platform.platform()}\n\n")

    f.write("=== Torch Info ===\n")
    f.write(f"Torch location: {torch.__file__}\n")
    f.write(f"Torch version: {torch.__version__}\n")
    f.write(f"CUDA version: {torch.version.cuda}\n")
    f.write(f"CUDA available: {torch.cuda.is_available()}\n")
    try:
        f.write(f"cuDNN Version: {torch.backends.cudnn.version()}\n")
    except Exception as e:
        f.write(f"cuDNN Version: Not available ({e})\n")
    f.write(f"Torch build:\n{torch.__config__.show()}\n\n")
    
    f.write("=== OpenCV Info ===\n")
    f.write(f"OpenCV version: {cv2.__version__}\n")
    f.write(f"OpenCV Build Info:\n{cv2.getBuildInformation()}\n")

print("System report generated at logs/system_report.txt")
