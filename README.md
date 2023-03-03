# SurRoL: An Open-source Reinforcement Learning Centered and dVRK Compatible Platform for Surgical Robot Learning

### [[Project Website]](https://med-air.github.io/SurRoL/)

ICRA'23 [Demonstration-Guided Reinforcement Learning with Efficient Exploration for Task Automation of Surgical Robot](https://arxiv.org/abs/2302.09772) <br>
ISMR'22 [Integrating artificial intelligence and augmented reality in robotic surgery: An initial dVRK study using a surgical education scenario](https://arxiv.org/abs/2201.00383) <br>
IROS'21 [SurRoL: An open-source reinforcement learning centered and dVRK compatible platform for surgical robot learning](https://arxiv.org/abs/2108.13035)


<p align="center">
   <img src="resources/img/surrol-overview.png" alt="SurRoL"/>
</p>

## Features

- [dVRK](https://github.com/jhu-dvrk/sawIntuitiveResearchKit/wiki) compatible [robots](./surrol/robots).
- [Gym](https://github.com/openai/gym) style [API](./surrol/gym) for reinforcement learning.
- Ten surgical-related [tasks](./surrol/tasks).
- Various object [assets](./surrol/assets).
- Based on [PyBullet]((https://github.com/bulletphysics/bullet3)) for physics simulation.

## Installation

The project is built on Ubuntu with Python 3.7,
[PyBullet](https://github.com/bulletphysics/bullet3),
[Gym 0.15.6](https://github.com/openai/gym/releases/tag/0.15.6),
and evaluated with [Baselines](https://github.com/openai/baselines),
[TensorFlow 1.14](https://www.tensorflow.org/install/pip).

### Prepare environment

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n surrol python=3.7 -y
    conda activate surrol
    ```

2. Install gym (slightly modified), tensorflow-gpu==1.14, baselines (modified).

### Install SurRoL

```shell
git clone https://github.com/jiaqixuac/surrol.git
cd surrol
pip install -e .
```

## Get started

The robot control API follows [dVRK](https://github.com/jhu-dvrk/dvrk-ros/tree/master/dvrk_python/src/dvrk)
(before "crtk"), which is compatible with the real-world dVRK robots.

You may have a look at the jupyter notebooks in [tests](./tests).
There are some test files for [PSM](./tests/test_psm.ipynb) and [ECM](./tests/test_ecm.ipynb),
that contains the basic procedures to start the environment, load the robot, and test the kinematics.

We also provide some [run files](./run) to evaluate the environments using baselines.

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
