# SurRoL v2

Under development...

## Installation

### Dependencies:

- PyBullet
- Gym
- pybullet_rendering
- Panda3D


### Prepare environment

```shell
conda create -n surrol python=3.7 -y
conda activate surrol
```
    
### Install SurRoL

```shell
git clone --recursive https://github.com/yuehaowang/SurRolv2.git
cd SurRolv2
pip install -e .
```

### Install Driver and Dependencies for Haptic Geomagic Touch

1. Install OpenHaptic Device Driver

    https://support.3dsystems.com/s/article/OpenHaptics-for-Linux-Developer-Edition-v34?language=en_US

2. Install SWIG -- https://www.swig.org/

3. Compile the Python API for Haptic Geomagic Touch
    ```shell
    cd tests
    bash setup_haptic.sh
    ```

### Test

```shell
# GUI
python tests/test_multiple_scenes.py
```
