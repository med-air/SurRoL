# Author(s): Jiaqi Xu
# Created on: 2020-11

"""
PSM wrapper
Refer to:
https://github.com/jhu-dvrk/dvrk-ros/blob/master/dvrk_python/src/dvrk/psm.py
https://github.com/jhu-dvrk/dvrk-ros/blob/7b3d48ca164755ccfc88028e15baa9fbf7aa1360/dvrk_python/src/dvrk/psm.py
https://github.com/jhu-dvrk/sawIntuitiveResearchKit/blob/master/share/kinematic/psm.json
https://github.com/jhu-dvrk/sawIntuitiveResearchKit/blob/master/share/tool/LARGE_NEEDLE_DRIVER_400006.json
https://github.com/jhu-dvrk/sawIntuitiveResearchKit/blob/4a8b4817ee7404b3183dfba269c0efe5885b41c2/share/arm/psm-large-needle-driver.json
"""
import os
import numpy as np
import pybullet as p

from surrol.robots.arm import Arm
from surrol.const import ASSET_DIR_PATH
from surrol.utils.pybullet_utils import (
    get_joint_positions,
)

LINKS = (
    'psm_base_link', 'psm_yaw_link', 'psm_pitch_end_link',  # -1, 0, 1
    'psm_main_insertion_link', 'psm_tool_roll_link',  # 2, 3
    'psm_tool_pitch_link', 'psm_tool_yaw_link',  # 4, 5
    'psm_tool_gripper1_link', 'psm_tool_gripper2_link',  # 6, 7
    'psm_tool_tip_link',  # 8
    'psm_pitch_back_link', 'psm_pitch_bottom_link',  # 9, 10
    'psm_pitch_top_link', 'psm_pitch_front_link',  # 11, 12
    'psm_remote_center_link',  # 13
    'psm_main_insertion_link_2',  # 14
    'psm_main_insertion_link_3',  # 15
)

# tooltip-offset; refer to .json
tool_T_tip = np.array([[0.0, -1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [-1.0, 0.0, 0.0, 0.0],
                       [0.0,  0.0, 0.0, 1.0]])

# Joint limit. TODO: dVRK config modified
TOOL_JOINT_LIMIT = {
    'lower': np.deg2rad([-91.0, -53.0,   0.0, -260.0, -80.0, -80.0, -20.0]),
    'upper': np.deg2rad([ 91.0,  53.0, 240.0,  260.0,  80.0,  80.0,  80.0]),
}
TOOL_JOINT_LIMIT['upper'][2] = 0.24  # prismatic joint (m)
# [-1.58824962, -0.9250245, 0.  , -4.53785606, -1.3962634, -1.3962634, -0.3490659]
# [ 1.58824962,  0.9250245, 0.24,  4.53785606,  1.3962634,  1.3962634,  1.3962634]


class Psm(Arm):
    NAME = 'PSM'
    URDF_PATH = os.path.join(ASSET_DIR_PATH, 'psm/psm.urdf')
    DoF = 6  # 6-dof arm
    JOINT_TYPES = ('R', 'R', 'P', 'R', 'R', 'R')
    EEF_LINK_INDEX = 5   # EEF link index
    TIP_LINK_INDEX = 8   # TIP link index
    RCM_LINK_INDEX = 13  # RCM link index
    A     = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0091])
    ALPHA = np.array([np.pi/2, -np.pi/2, np.pi/2, 0.0, -np.pi/2, -np.pi/2])
    D     = np.array([0.0, 0.0, -0.4318, 0.4162, 0.0, 0.0])
    THETA = np.array([np.pi/2, -np.pi/2, 0.0, 0.0, -np.pi/2, -np.pi/2])

    def __init__(self, pos=(0., 0., 0.1524), orn=(0., 0., 0., 1.),
                 scaling=1.):
        super(Psm, self).__init__(self.URDF_PATH, pos, orn,
                                  TOOL_JOINT_LIMIT, tool_T_tip, scaling)

        self._jaw_angle = self.get_current_jaw_position() / 2

    def get_current_jaw_position(self) -> float:
        """ Get the current angle of the jaw in radians. """
        positions = get_joint_positions(self.body, (6, 7))
        return -positions[0] + positions[1]

    def close_jaw(self) -> bool:
        """ Close the tool jaw. """
        # TODO: use 0 instead of -20; may need to add self collision
        return self.move_jaw(np.deg2rad(0.0))

    def open_jaw(self) -> bool:
        """ Open the tool jaw. """
        return self.move_jaw(np.deg2rad(80.0))

    def move_jaw(self, angle_radian: float) -> bool:
        """ Set the jaw tool to angle_radian in radians without actual move. """
        angle = angle_radian / 2
        self._jaw_angle = angle
        for joint in (6, 7):
            position = - angle if joint == 6 else angle
            p.setJointMotorControl2(self.body,
                                    joint,  # jaw joints
                                    p.POSITION_CONTROL,
                                    targetPosition=position,
                                    force=2.)  # TODO: not sure about the force, need tune
        return True

    def _get_joint_positions_all(self, abs_input):
        """ With the consideration of parallel mechanism constraints and other redundant joints.
        """
        positions = get_joint_positions(self.body, self.joints)
        joint_positions = [
            abs_input[0], abs_input[1],  # 0, 1
            abs_input[2] * self.scaling, abs_input[3],  # 2, 3
            abs_input[4], abs_input[5],  # 4, 5
            -self._jaw_angle, self._jaw_angle,  # 6, 7, important to set jaw_angle if holding a object
            positions[8],  # 8 (0.0)
            abs_input[1], -abs_input[1],  # 9, 10
            -abs_input[1], abs_input[1],  # 11, 12
            positions[13],  # 13 (0.0)
            positions[14], positions[15],  # 14 (0.0), 15 (0.0)
        ]
        return joint_positions

    def _set_collision(self):
        group, mask = 0, 0
        p.setCollisionFilterGroupMask(self.body, 8, group, mask)  # tip
        p.setCollisionFilterGroupMask(self.body, 13, group, mask)  # RCM

    def _set_constraint(self):
        """
        Not sure whether it will affect the physics simulation.
        Empirically, it's not good for grasping objects.
        The parameters refer to
        https://github.com/WPI-AIM/ambf/blob/ambf-1.0/ambf_models/descriptions/multi-bodies/robots/blender-psm.yaml
        """
        # gear joint: gripper constraint
        c = p.createConstraint(self.body,
                               6,
                               self.body,
                               7,
                               jointType=p.JOINT_GEAR,
                               jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1)

        # # p2p joint: top-end
        # c = p.createConstraint(self.body,
        #                        11,
        #                        self.body,
        #                        1,
        #                        jointType=p.JOINT_POINT2POINT,
        #                        jointAxis=[0, 0, 1],
        #                        parentFramePosition=[0.516 - 0.31246, 0.0 - -0.00056566, 0.0 - 0.0],
        #                        # offset - initial xyz
        #                        childFramePosition=[0.03906 - 0.0442293, 0.18001 - 0.27985, 0.0 - 0.0])
        # p.changeConstraint(c, gearRatio=-1, maxForce=1000)
        #
        # # p2p joint: bottom-end
        # c = p.createConstraint(self.body,
        #                        10,
        #                        self.body,
        #                        1,
        #                        jointType=p.JOINT_POINT2POINT,
        #                        jointAxis=[0, 0, 1],
        #                        parentFramePosition=[0.516 - 0.25683, 0.0 - -0.010348, 0.0 - 0.0],
        #                        # offset - initial xyz
        #                        childFramePosition=[0.04295 - 0.0442293, 0.14372 - 0.27985, 0.0 - 0.0])
        # p.changeConstraint(c, gearRatio=-1, maxForce=1000)
        #
        # # p2p joint: front-bottom
        # c = p.createConstraint(self.body,
        #                        12,
        #                        self.body,
        #                        10,
        #                        jointType=p.JOINT_POINT2POINT,
        #                        jointAxis=[0, 0, 1],
        #                        parentFramePosition=[0.15 - 0.096269, 0.0 - 0.0, -0.0002 - 0.047551],
        #                        childFramePosition=[0.096 - 0.25683, 0.0 - -0.010348, -0.0002 - 0.0])
        # p.changeConstraint(c, gearRatio=-1, maxForce=1000)


class Psm1(Psm):
    NAME = 'PSM1'


class Psm2(Psm):
    NAME = 'PSM2'

    def __init__(self, pos=(0., 0., 0.1524), orn=(0., 0., 0., 1.),
                 scaling=1.):
        super(Psm2, self).__init__(pos, orn, scaling)

        # change the end link color
        p.changeVisualShape(self.body, 1, rgbaColor=(20. / 255, 156. / 255, 20. / 255, 1.))
