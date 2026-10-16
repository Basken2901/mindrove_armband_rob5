# Mindrove Armband EMG Interface
![Platform](https://img.shields.io/badge/platform-Linux%20(Ubuntu%2024.04)-orange?logo=linux)
![ROS 2](https://img.shields.io/badge/ROS%202-Jazzy-blue?logo=ros)
![Status](https://img.shields.io/badge/status-active%20research-brightgreen)

This projects purpose is to record data and control the mouse or other types of interfaces via. EMG signals distributed from the Mindrove armband.

### Create a virtual environment for packages
This is already made into an .sh file that automatically makes an virtual environment, that downloads the needed packages.

1. Enter the setup folder
```
cd mindrove_armband_rob5/setup
```

2. make the .sh file executable
```
chmod +x install_mindrove.sh
```

3. Execute it by using
```
./install_mindrove.sh
```

### Starting the sampling
It is expected that the Mindrove armband is already connected via the internet.

1. Enter the virtual environtment
```
source ~/mindrove_venv/bin/activate
```

2. Enter the repository
```
cd mindrove_armband_rob5
```

3. Run colcon with the virtual environment's Python, then source the workspace
´´´
python "$(command -v colcon)" build
´´´
´´´
source install/setup.bash
```

4. Run the sampling
```
ros2 run data_recording mindrove_node
```
