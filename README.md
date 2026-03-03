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
| A | Start recording |
| B | Save demonstration |
| X | Reset episode |
| Y | Pause recording |
| Right stick | Discard episode |