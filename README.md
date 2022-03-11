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

### Test

```shell
# GUI
python tests/test_multiple_scenes.py
```

