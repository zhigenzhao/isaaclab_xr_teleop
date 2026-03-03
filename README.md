# XR Teleoperation for IsaacLab
## Installation
1. **Cloning the repo**
```
git clone https://github.com/zhigenzhao/isaaclab_xr_teleop.git
cd isaaclab_xr_teleop
git submodule update --init --recursive
```
2. **Initialize Conda Env**
```bash
conda create -n isaaclab_teleop python=3.11
conda activate isaaclab_teleop
conda install -c conda-forge libstdcxx-ng -y
```
3. **Installing Isaaclab**
```bash
# installing isaacsim
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# installing isaaclab
cd submodules/IsaacLab
./isaaclab.sh --install
cd ../..
```
4. **Installing xrobotoolkit_sdk**
```bash
cd submodules/XRoboToolkit-PC-Service-Pybind
bash setup_ubuntu.sh
cd ../..
```
5. **Installing isaaclab_xr_teleop**
```bash
pip install -e .
```
6. **Install XRoboToolkit-PC-Service**  
- Download [deb package for ubuntu 22.04](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb), or build from the [repo source](https://github.com/XR-Robotics/XRoboToolkit-PC-Service).
- The XRoboToolkit-PC-Service has been tested on ubuntu 24.04. Download [deb package for ubuntu 24.04](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb)
- To install, use command
    ```bash
    sudo dpkg -i XRoboToolkit-PC-Service_1.0.0_ubuntu_22.04_amd64.deb
    ```
    or
    ```bash
    sudo dpkg -i XRoboToolkit-PC-Service_1.0.0_ubuntu_24.04_amd64.deb
    ```
- Note that the XRoboToolkit-PC-Service App should be turned on after every reboot for the headset to be connected.

7. **Install the XR App on Headset**
- Turn on developer mode on Pico 4 Ultra headset first ([Enable developer mode on Pico 4 Ultra](https://developer.picoxr.com/ja/document/unreal/test-and-build/)), and make sure that [adb](https://developer.android.com/tools/adb) is installed properly.
- Download [XRoboToolkit-PICO-1.1.1.apk](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk) on a PC with adb installed. <sup>[[Other Versions](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases)]</sup>
- To install apk on the headset, use command
      ```bash
      adb install -g XRoboToolkit-PICO-1.1.1.apk
      ```
- In the VR app, connect to the PC via the IP address.
- Under ```Tracking```, select head, controller, and send. You should be able to see the headset and controller animation moving in the PC app.
- For more information on XRoboToolkit, see [here](https://github.com/XR-Robotics).
## Examples

### Franka Cube Stacking
Record teleoperation demonstrations for stacking cubes with a Franka Panda arm (relative SE3 control):
```bash
python -m isaaclab_xr_teleop.tasks.franka_stack.record \
    --task Isaac-Stack-Cube-Franka-IK-Rel-XR-v0 \
    --teleop_device xr_controller --device cpu
```

### G1 Humanoid Pick-Place
Record teleoperation demonstrations for pick-and-place with a Unitree G1 humanoid (dual-arm Mink IK control):
```bash
python -m isaaclab_xr_teleop.tasks.g1_pick_place.record \
    --task Isaac-PickPlace-G1-InspireFTP-XR-v0 \
    --teleop_device xr_controller --enable_cameras --device cpu
```

### Common Arguments
| Argument | Default | Description |
|---|---|---|
| `--step_hz` | 50 | Environment stepping rate in Hz |
| `--num_demos` | 0 | Number of demonstrations to record (0 = infinite) |
| `--num_success_steps` | 10 | Continuous success steps before auto-save |
| `--dataset_file` | `./datasets/dataset.hdf5` | Output path for recorded demos |

### VR Controller Buttons
| Button | Action |
|---|---|
| Left/Right Grip| Activate Left/Right Arm |
| Left/Right Trigger | Close Left/Right Gripper |
| A | Start recording |
| B | Save demonstration |
| X | Reset episode |
| Y | Pause recording |
| Right stick | Discard episode |

## Data Collection

### Recording Workflow

1. **Start the recording script** for your task (see [Examples](#examples) above). Ensure the XRoboToolkit-PC-Service is running and the VR headset is connected.

2. **Teleoperate the robot** using the VR controllers:
   - Hold **Grip** (left/right) to activate the corresponding arm.
   - Use **Trigger** (left/right) to close the gripper.

3. **Control the recording** with the VR buttons:
   - Press **A** to start recording the current episode.
   - Press **Y** to pause recording without discarding the buffer.
   - Press **B** to save the episode as a successful demonstration.
   - Press **X** to reset the episode (clears the buffer and resets the scene).
   - Push **Right Stick** to discard the current episode without saving.

4. **Auto-save:** If the task has a success condition (e.g., cube stacked), the episode is automatically saved after `--num_success_steps` consecutive success steps (default: 10).

5. **Termination:** The script exits automatically after `--num_demos` successful episodes are collected. If `--num_demos` is 0 (default), recording runs indefinitely until you press `Ctrl+C`.

### Example: Collect 50 Franka Demos

```bash
python -m isaaclab_xr_teleop.tasks.franka_stack.record \
    --task Isaac-Stack-Cube-Franka-IK-Rel-XR-v0 \
    --teleop_device xr_controller \
    --dataset_file ./datasets/franka_stack.hdf5 \
    --num_demos 50 \
    --num_success_steps 10 \
    --step_hz 50 \
    --device cpu
```

### Output Format

Demonstrations are saved in **HDF5** format. The output file is timestamped automatically (e.g., `franka_stack_20260303_142530.hdf5`) and placed in the directory derived from `--dataset_file`. Only successfully completed episodes are exported.

To read the collected data:
```python
import h5py

with h5py.File("datasets/franka_stack_20260303_142530.hdf5", "r") as f:
    for episode_id in f.keys():
        episode = f[episode_id]
        print(f"Episode {episode_id}: {len(episode['actions'])} steps")
```