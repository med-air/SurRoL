## Assets

This folder contains the meshes and [URDF](http://wiki.ros.org/urdf/Tutorials) files of the objects. dVRK meshes are
modified from [AMBF](https://github.com/WPI-AIM/ambf). Note that PyBullet computes the forward and inverse kinematics
when the links are in sequential order, which is not the property in the original AMBF files. So we modify the link
frames to enable this feature. Besides, most of the objects are modeled using [Blender](https://www.blender.org/).
Furthermore, to make the physics simulation more stable, especially if there are physical contacts, we
use [V-HACD](https://github.com/kmammou/v-hacd) to decompose the visual meshes into convex parts for the collision
computation.