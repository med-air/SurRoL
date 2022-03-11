from logging import root
import os
import sys
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess


def check_directories():
    root_files = os.listdir('.')
    assert 'setup.py' in root_files and 'submodules' in root_files, 'Installation NOT in the root directory of SurRol.'
    
    submodules = os.listdir('./submodules')
    assert 'bullet3' in submodules, 'Submodule `bullet3` not found.'
    assert 'pybullet_rendering' in submodules, 'Submodule `pybullet_rendering` not found.'

def install_submodules():
    # Check prerequisites of project directories
    print('=== Check project directories')
    check_directories()
    print('  -- done')

    # Install submodules
    print('\n=== Install submodules')
    os.chdir('./submodules/bullet3')
    bullet_root_dir = os.path.realpath('.')
    subprocess.check_call([os.path.abspath(sys.executable), "setup.py", "install"])
    print('  -- pybullet installed')
    os.chdir('../pybullet_rendering')
    subprocess.check_call([os.path.abspath(sys.executable), "setup.py", "install", "--bullet_dir", bullet_root_dir])
    os.chdir('../../')
    print('  -- pybullet_rendering installed')

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        install_submodules()

class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        install_submodules()


if __name__ == '__main__':
    setup(
        name='surrol',
        version='0.2.0',
        description='SurRoL: An Open-source Reinforcement Learning Centered and '
                    'dVRK Compatible Platform for Surgical Robot Learning',
        author='Med-AIR@CUHK',
        keywords='simulation, medical robotics, dVRK, reinforcement learning',
        packages=[
            'surrol',
        ],
        python_requires = '>=3.7',
        install_requires=[
            "gym>=0.15.6",
            "numpy>=1.21.1",
            "scipy",
            "pandas",
            "imageio",
            "imageio-ffmpeg",
            "opencv-python",
            "roboticstoolbox-python",
            "sympy",
            "panda3d==1.10.11",
            "trimesh",
        ],
        cmdclass = {
            'install': PostInstallCommand,
            'develop': PostDevelopCommand
        },
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
    

