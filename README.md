# SurRoL-v2


<p align="center">
   <img src="resources/img/overview.png" width="95%" height="95%" alt="SurRoL"/>
</p>

- [Human-in-the-loop Embodied Intelligence with Interactive Simulation Environment for Surgical Robot Learning](https://arxiv.org/abs/2301.00452)

## Features

- [dVRK](https://github.com/jhu-dvrk/sawIntuitiveResearchKit/wiki) compatible [robots](./surrol/robots).
- [Gym](https://github.com/openai/gym) style [API](./surrol/gym) for reinforcement learning.
- 14 surgical-related [tasks](./surrol/tasks).
- Various object [assets](./surrol/assets).
- Based on [PyBullet](https://github.com/bulletphysics/bullet3) for physics simulation.
- Based on [Panda3D](https://www.panda3d.org/) for GUI and scene rendering.
- Allow human interaction with [Touch Haptic Device](https://www.3dsystems.com/haptics-devices/touch).
- Extenable designs which allows customization as needed.

## Installation

The project is built on Ubuntu with Python 3.7.

### Prepare environment

Create a conda virtual environment and activate it.

 ```shell
 conda create -n surrol python=3.7 -y
 conda activate surrol
 ```

### Install SurRoL

   ```shell
   git clone --recursive https://github.com/med-air/SurRoL.git
   cd SurRoL
   git checkout SurRoL_v2
   pip install -e .
   ```

### Install Driver and Dependencies for Touch Haptic Device

1. Install [OpenHaptic Device Driver](https://support.3dsystems.com/s/article/OpenHaptics-for-Linux-Developer-Edition-v34?language=en_US)    

2. Setup Device Name for Identification.

     Run the "Touch_Setup" software provided by the OpenHaptic Device Driver. 
     <p align="left">
      <img src="resources/img/SetupTouch.png" width="30%" height="30%" alt="SurRoL"/>
      </p>
     Set the right device name as "right" and set the left device name as "left".

3. Install SWIG (>=4.0.2) -- https://www.swig.org/

4. Compile the Python API of Touch Haptic Device for SurRoL-v2
    ```shell
    bash setup_haptic.sh
    ```

## Get started

The robot control API follows [dVRK](https://github.com/jhu-dvrk/dvrk-ros/tree/master/dvrk_python/src/dvrk)
(before "crtk"), which is compatible with the real-world dVRK robots.

You may have a look at the jupyter notebooks in [tests](./tests).
There are some test files for [PSM](./tests/test_psm.ipynb) and [ECM](./tests/test_ecm.ipynb),
that contains the basic procedures to start the environment, load the robot, and test the kinematics.

We also provide some [run files](./run) to evaluate the environments using baselines.

To start the SurRoL-v2 GUI, run the following command:
```shell
# GUI
python tests/test_multiple_scenes.py
```
You should see the following windows:
<p align="center">
   <img src="resources/img/GUI.png" width="95%" height="95%" alt="SurRoL"/>
</p>

## Citation

If you find the paper or the code helpful to your research, please cite the project.

```
@inproceedings{xu2021surrol,
  title={SurRoL: An Open-source Reinforcement Learning Centered and dVRK Compatible Platform for Surgical Robot Learning},
  author={Xu, Jiaqi and Li, Bin and Lu, Bo and Liu, Yun-Hui and Dou, Qi and Heng, Pheng-Ann},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2021},
  organization={IEEE}
}

@article{long2023human,
  title={Human-in-the-loop Embodied Intelligence with Interactive Simulation Environment for Surgical Robot Learning},
  author={Long, Yonghao and Wei, Wang and Huang, Tao and Wang, Yuehao and Dou, Qi},
  journal={arXiv preprint arXiv:2301.00452},
  year={2023}
}
```
## License

SurRoL is released under the [MIT license](LICENSE).


## Acknowledgement

The code is built with the reference of [dVRK](https://github.com/jhu-dvrk/sawIntuitiveResearchKit/wiki),
[AMBF](https://github.com/WPI-AIM/ambf),
[dVRL](https://github.com/ucsdarclab/dVRL),
[RLBench](https://github.com/stepjam/RLBench),
[Decentralized-MultiArm](https://github.com/columbia-ai-robotics/decentralized-multiarm),
[Ravens](https://github.com/google-research/ravens), etc.


## Contact
For any questions, please feel free to email <a href="mailto:qidou@cuhk.edu.hk">qidou@cuhk.edu.hk</a>
