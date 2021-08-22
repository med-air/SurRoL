### dVRK PSM

<p align="center">
   <img src="blender_files/psm.png" alt="SurRoL PSM"/>
</p>

The psm_ori.urdf is generated from [WPI-AIM PSM](https://github.com/WPI-AIM/dvrk_env/tree/master/dvrk_description/psm).

To generate a .urdf file, run the following command on Linux command-line

```shell
rosrun xacro xacro psm_launch.urdf.xacro > psm.urdf
```

The meshes are modified from [AMBF PSM](https://github.com/WPI-AIM/ambf/tree/ambf-1.0/ambf_models/meshes/dvrk/psm).

We rebuild the link frame and structural topology such that the six DoFs are in a chain manner for built-in inverse
kinematics calculation.
Please compare the [modified](./blender_files/psm_wpi_modified.blend)
with [original](./blender_files/psm_wpi_ori.blend) Blender files to see the details. 
We then modify the D-H parameters accordingly (there also exist some errors in the original .xacro file). 
Please also refer to
[AMBF model descriptions](https://github.com/WPI-AIM/ambf/tree/ambf-1.0/ambf_models),
[dVRK model](https://github.com/jhu-dvrk/dvrk-ros/tree/master/dvrk_model/model).

**Note:** It is non-trivial to get the dynamics right (e.g., collision shapes, friction, scaling, etc.) to enable stable
physical interaction.

### TODO:

- Check the joint limits.
- Need to modify the dynamics parameters.
