# Author(s): Jiaqi Xu
# Created on: 2020-11

"""
Arm wrapper
Refer to:
https://github.com/jhu-dvrk/dvrk-ros/blob/master/dvrk_python/src/dvrk/arm.py
https://github.com/jhu-dvrk/dvrk-ros/blob/7b3d48ca164755ccfc88028e15baa9fbf7aa1360/dvrk_python/src/dvrk/arm.py
"""
from typing import Union
import numpy as np
import pybullet as p
import roboticstoolbox as rtb

from surrol.utils.pybullet_utils import (
    get_joints,
    get_joint_positions,
    set_joint_positions,
    get_link_pose,
    forward_kinematics,
    inverse_kinematics,
    wrap_angle,
)
from surrol.utils.robotics import (
    get_matrix_from_pose_2d,
    get_pose_2d_from_matrix,
)


class Arm(object):
    """
    Arm wrapper
    Assume the controlled joints are in serial for the first self.DoF joints.
    """
    NAME = 'Arm'
    URDF_PATH = None  # the urdf path to load
    DoF = 0  # int
    JOINT_TYPES = ('R',)  # 'R' or 'P' for revolute or prismatic
    EEF_LINK_INDEX = DoF - 1  # EEF link index
    TIP_LINK_INDEX = DoF - 1  # TIP link index
    RCM_LINK_INDEX = 0        # RCM link index
    # D-H parameters
    A     = np.array([0.0])
    ALPHA = np.array([0.0])
    D     = np.array([0.0])
    THETA = np.array([0.0])

    def __init__(self, urdf_file,
                 pos=(0., 0., 0.), orn=(0., 0., 0., 1.),
                 limits=None, tool_T_tip=np.eye(4), scaling=1.):
        """
        :param urdf_file: URDF fileName.
        :param pos: basePosition.
        :param orn: baseOrientation in quaternion.
        """
        # should connect to the PyBullet server first
        self.body = p.loadURDF(urdf_file,
                               np.array(pos) * scaling, orn,
                               useFixedBase=True, globalScaling=scaling,
                               flags=p.URDF_MAINTAIN_LINK_ORDER)  # self.collision=True is not suitable
        self.joints = get_joints(self.body)
        self._position_joint_desired = np.zeros(self.DoF)
        self.limits = limits
        self.tool_T_tip = tool_T_tip  # tool_T_tip offset
        self.scaling = scaling  # scaling factor

        # update RCM pose and related transformations
        self.wTr, self.rTw = None, None
        self.update_rcm_pose()

        # update EEF and TIP related transformations
        self.eTt, self.tTe = None, None
        self.update_tip_pose()

        self._set_collision()
        # self._add_constraint()  # have effect when the joint positions are not set

        # use roboticstoolbox to calculate the forward and inverse kinematics quickly
        links = []
        for i in range(self.DoF):
            # DH parameters
            if self.JOINT_TYPES[i] == 'R':
                links.append(rtb.RevoluteMDH(alpha=self.ALPHA[i], a=self.A[i], d=self.D[i], offset=self.THETA[i]))
            else:
                links.append(rtb.PrismaticMDH(alpha=self.ALPHA[i], a=self.A[i], theta=self.THETA[i], offset=self.D[i]))
        self.robot = rtb.DHRobot(links, name=self.NAME)

    def get_current_position(self) -> np.ndarray:
        """ Get the 'current cartesian position' of the arm (RCM frame).
        Return 4*4 matrix. """
        pose_world = forward_kinematics(self.body, eef_link=self.DoF - 1)
        pose_rcm = self.pose_world2rcm(pose_world, 'matrix')
        return pose_rcm

    def get_current_joint_position(self) -> list:
        """ Get the 'current joint position' of the arm. """
        joint_positions = get_joint_positions(self.body, self.joints[:self.DoF])
        for i in range(self.DoF):
            if self.JOINT_TYPES[i] == 'P':
                # get the unscaled joint position
                joint_positions[i] /= self.scaling
        return joint_positions

    def get_desired_joint_position(self):
        """ Get the 'desired joint position' of the arm. """
        return self._position_joint_desired

    def get_joint_number(self) -> int:
        """ Get the number of joints on the arm specified. """
        return self.DoF

    def dmove_joint(self, delta_pos: [list, np.ndarray]) -> [bool, np.ndarray]:
        """ Incremental move in joint space.
        """
        if not isinstance(delta_pos, np.ndarray):
            delta_pos = np.array(delta_pos)
        abs_pos = np.array(self.get_current_joint_position())  # or self._position_joint_desired ?
        abs_pos += delta_pos
        return self.move_joint(abs_pos)

    def dmove_joint_one(self, delta_pos: float, indices: int) -> bool:
        """ Incremental index move of 1 joint in joint space.
        """
        return self.dmove_joint_some(np.array([delta_pos]), np.array([indices]))

    def dmove_joint_some(self, delta_pos: np.ndarray, indices: np.ndarray) -> bool:
        """ Incremental index move of a series of joints in joint space.
        """
        if not len(delta_pos) == len(indices):
            return False
        abs_pos = np.array(self.get_current_joint_position())
        for i in range(len(indices)):
            abs_pos[indices[i]] += delta_pos[i]
        return self.move_joint(abs_pos)

    def move_joint(self, abs_input: [list, np.ndarray]) -> [bool, np.ndarray]:
        """
        Absolute move in joint space.
        Set desired joint positions without actual physical move (need pybullet to step).
        :param abs_input: the absolute translation you want to make (in joint space).
        :return: whether or not able to reach the given input.
        """
        if not self._check_joint_limits(abs_input):
            return False
        self._position_joint_desired = np.copy(abs_input)
        joint_positions = self._get_joint_positions_all(abs_input)
        p.setJointMotorControlArray(self.body,
                                    self.joints,
                                    p.POSITION_CONTROL,
                                    targetPositions=joint_positions,
                                    targetVelocities=[0.] * len(joint_positions))
        return abs_input

    def move(self, abs_input: np.ndarray, link_index=None) -> [bool, np.ndarray]:
        """
        Absolute translation in Cartesian space (RCM frame).
        Set target joint positions without actual physical move (need pybullet to step).
        :param abs_input: the absolute translation you want to make (in Cartesian space, tip_T_rcm, 4*4).
        :param link_index: the index for the link to compute inverse kinematics; should be consistent with dVRK.
        :return: whether or not able to reach the given input.
        """
        assert abs_input.shape == (4, 4)
        if link_index is None:
            # default link index is the DoF
            link_index = self.EEF_LINK_INDEX
        pose_world = self.pose_rcm2world(abs_input, 'tuple')
        # joints_inv = np.array(inverse_kinematics(self.body, self.EEF_LINK_INDEX,
        #                                          pose_world[0], pose_world[1]))
        joints_inv = self.inverse_kinematics(pose_world, link_index)
        return self.move_joint(joints_inv)

    def update_rcm_pose(self):
        """ Update the world_T_rcm (wTr) and rcm_T_world (rTw) transformation matrix.
        """
        positions = get_joint_positions(self.body, self.joints)  # dummy positions; not affect rcm pose
        # RCM pose in the world frame
        world_pose_rcm = forward_kinematics(self.body, self.joints, positions, self.RCM_LINK_INDEX)
        self.wTr = get_matrix_from_pose_2d(world_pose_rcm)  # world_T_rcm
        self.rTw = np.linalg.inv(self.wTr)

    def update_tip_pose(self):
        """
        Update the eef_T_tip (eTt) and tip_T_eef (tTe) transformation matrix.
        The EEF link can be either the same link of Tip or the other link.
        """
        world_pose_eef = get_link_pose(self.body, self.EEF_LINK_INDEX)
        wTe = get_matrix_from_pose_2d(world_pose_eef)  # world_T_eef
        world_pose_eef = get_link_pose(self.body, self.TIP_LINK_INDEX)
        wTt = get_matrix_from_pose_2d(world_pose_eef)  # world_T_tip
        self.eTt = np.matmul(np.linalg.inv(wTe), wTt)
        self.tTe = np.linalg.inv(self.eTt)

    def pose_rcm2world(self, pose: Union[tuple, list, np.ndarray], option=None):
        """
        PyBullet helper function to transform pose from the RCM frame to the world frame.
        With tool-tip offset.
        :param pose: offset 'tip' pose in the RCM frame; normalized by the scaling factor.
        :param option: which output type of transformed pose should be, 'tuple' or 'matrix'.
        :return: pose in the world frame.
        """
        # rcm_T_tip -> rcm_T_tool
        pose_rcm = self._pose_transform(pose, np.linalg.inv(self.tool_T_tip), premultiply=False)
        pose_rcm[0: 3, 3] *= self.scaling  # recover the original size
        # rcm_T_tool -> world_T_tool
        pose_world = self._pose_transform(pose_rcm, self.wTr)
        if option == 'tuple' or (option is None and isinstance(pose, (tuple, list))):
            pose_world = get_pose_2d_from_matrix(pose_world)
        return pose_world

    def pose_world2rcm(self, pose: Union[tuple, list, np.ndarray], option=None):
        """
        PyBullet helper function to transform pose from the world frame to the RCM frame.
        With tool-tip offset.
        :param pose: 'tool' (eef) pose in the world frame.
        :param option: which type of transformed pose should be, 'tuple' or 'matrix'.
        :return: pose in the RCM frame; normalized by the scaling factor.
        """
        # world_T_tool -> rcm_T_tool
        pose_rcm = self._pose_transform(pose, self.rTw, premultiply=True)
        # rcm_T_tool -> rcm_T_tip
        pose_rcm = np.matmul(pose_rcm, self.tool_T_tip)
        pose_rcm[0: 3, 3] /= self.scaling  # scaled
        if option == 'tuple' or (option is None and isinstance(pose, (tuple, list))):
            pose_rcm = get_pose_2d_from_matrix(pose_rcm)
        return pose_rcm

    def pose_tip2eef(self, pose: Union[tuple, list, np.ndarray], option=None):
        """
        Helper function to transform the tip pose given in the world frame to the eef pose.
        :param pose: actual tip pose in the any frame.
        :param option: which type of transformed pose should be, 'tuple' or 'matrix'.
        :return: pose in the RCM frame; normalized by the scaling factor.
        """
        # any_T_tip -> any_T_eef
        pose_eef = self._pose_transform(pose, self.tTe, premultiply=False)
        if option == 'tuple' or (option is None and isinstance(pose, (tuple, list))):
            pose_eef = get_pose_2d_from_matrix(pose_eef)
        return pose_eef

    def reset_joint(self, abs_input: [list, np.ndarray]) -> [bool, np.ndarray]:
        """
        Helper function for PyBullet initial reset.
        Not recommend to use during simulation.
        """
        if not self._check_joint_limits(abs_input):
            return
        joint_positions = self._get_joint_positions_all(abs_input)
        set_joint_positions(self.body, self.joints, joint_positions)
        return self.move_joint(abs_input)

    def inverse_kinematics(self, pose_world: tuple, link_index: None) -> np.ndarray:
        """
        Compute the inverse kinematics using PyBullet built-in methods.
        Given the pose in the world frame, output the joint positions normalized by self.scaling.
        """
        if link_index is None:
            link_index = self.DoF - 1
        joints_inv = p.calculateInverseKinematics(
            bodyUniqueId=self.body,
            endEffectorLinkIndex=link_index,
            targetPosition=pose_world[0],  # inertial pose, not joint pose
            targetOrientation=pose_world[1],
            lowerLimits=self.limits['lower'][:self.DoF],
            upperLimits=self.limits['upper'][:self.DoF],
            jointRanges=self.limits['upper'][:self.DoF] - self.limits['lower'][:self.DoF],
            restPoses=[0] * self.DoF,
            residualThreshold=1e-9,  # can tune
            maxNumIterations=200
        )
        # joints_inv = inverse_kinematics(self.body, link_index, pose_world[0], pose_world[1])
        joints_inv = np.array(joints_inv)
        for i in range(self.DoF):
            if self.JOINT_TYPES[i] == 'P':
                joints_inv[i] /= self.scaling
        return wrap_angle(joints_inv[:self.DoF])

    def get_jacobian_spatial(self, qs=None) -> np.ndarray:
        """
        Calculate the Jacobian matrix in the base (world?rcm) frame using the Peter Corke toolbox.
        (PyBullet uses the initial frame instead of the joint frame, not sure).
        return Jacobian matrix in shape (6, DoF).
        """
        if qs is None:
            qs = self.get_current_joint_position()
        return self.robot.jacob0(qs)

    def _check_joint_limits(self, abs_input: [list, np.ndarray]):
        """ Check if the joint set is within the joint limits.
        """
        assert len(abs_input) == self.DoF, "The number of joints should match the arm DoF."
        if not np.all(np.bitwise_and(abs_input >= self.limits['lower'][:self.DoF],
                                     abs_input <= self.limits['upper'][:self.DoF])):
            print("Joint position out of valid range!")
            print("Set joint:", abs_input)
            return False
        return True

    def _get_joint_positions_all(self, abs_input: [list, np.ndarray]):
        """ With the consideration of parallel mechanism constraints and other redundant joints.
        """
        return np.copy(abs_input)

    @staticmethod
    def _pose_transform(pose, mat: np.ndarray, premultiply=True) -> np.ndarray:
        """
        :param pose: tuple (position (3), orientation (4)) or matrix (4*4).
        :param mat: transformation matrix.
        :param premultiply: premultiply or postmultiply the mat.
        :return: pose in the transformed frame.
        """
        if isinstance(pose, (tuple, list)):
            pose_ori = get_matrix_from_pose_2d(pose)
        else:
            pose_ori = pose.copy()
        if premultiply:
            pose_tf = np.matmul(mat, pose_ori)
        else:
            pose_tf = np.matmul(pose_ori, mat)
        return pose_tf

    def _set_collision(self):
        """ Set collision groups.
        """
        pass

    def _set_constraint(self):
        """ Set if there is any constraint to maintain the parallel link.
        """
        pass
