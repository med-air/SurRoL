from setuptools import setup

if __name__ == '__main__':
    setup(
        name='surrol',
        version='0.1.0',
        description='SurRoL: An Open-source Reinforcement Learning Centered and '
                    'dVRK Compatible Platform for Surgical Robot Learning',
        author='Med-AIR@CUHK',
        keywords='simulation, medical robotics, dVRK, reinforcement learning',
        packages=[
            'surrol',
        ],
        install_requires=[
            "gym>=0.15.6",
            "pybullet>=3.0.7",
            "numpy>=1.21.1",
            "scipy",
            "pandas",
            "imageio",
            "imageio-ffmpeg",
            "opencv-python",
            "roboticstoolbox-python",
        ],
        extras_require={
            # optional dependencies, required by evaluation, test, etc.
            "all": [
                "tensorflow-gpu==1.14",
                "baselines",
                "mpi4py",  # important for ddpg
                "ipython",
                "jupyter",
            ]
        }
    )
