# Author(s): Jiaqi Xu
# Created on: 2020-11

"""
PSM wrapper
Refer to:
https://github.com/jhu-dvrk/dvrk-ros/blob/master/dvrk_python/src/dvrk/ecm.py
https://github.com/jhu-dvrk/dvrk-ros/blob/7b3d48ca164755ccfc88028e15baa9fbf7aa1360/dvrk_python/src/dvrk/ecm.py
https://github.com/jhu-dvrk/sawIntuitiveResearchKit/blob/master/share/kinematic/ecm.json
https://github.com/jhu-dvrk/sawIntuitiveResearchKit/blob/4a8b4817ee7404b3183dfba269c0efe5885b41c2/share/arm/ecm-straight.json
"""
import os
import numpy as np
import pybullet as p

from surrol.robots.arm import Arm
from surrol.const import ASSET_DIR_PATH
from surrol.utils.pybullet_utils import (
    get_joint_positions,
    get_link_pose,
    render_image
)

# Rendering width and height
RENDER_HEIGHT = 256
RENDER_WIDTH = 256
FoV = 60

LINKS = (
    'ecm_base_link', 'ecm_yaw_link', 'ecm_pitch_end_link',  # -1, 0, 1
    'ecm_main_insertion_link', 'ecm_tool_link',  # 2, 3
    'ecm_end_link',  # 4
    'ecm_tip_link',  # 5
    'ecm_pitch_front_link',  # 6
    'ecm_pitch_bottom_link', 'ecm_pitch_top_link',  # 7, 8
    'ecm_pitch_back_link',  # 9
    'ecm_remote_center_link',  # 10
)

# tooltip-offset; refer to .json
tool_T_tip = np.array([[0.0, 1.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])

# Joint limits. No limits in the .json. TODO: dVRK config modified
TOOL_JOINT_LIMIT = {
    'lower': np.deg2rad([-90.0, -45.0,   0.0, -np.inf]),  # not sure about the last joint
    'upper': np.deg2rad([ 90.0,  66.4, 254.0,  np.inf]),
}
TOOL_JOINT_LIMIT['lower'][2] = -0.01  # allow small tolerance
TOOL_JOINT_LIMIT['upper'][2] = 0.254  # prismatic joint (m); not sure, from ambf
# [-1.57079633, -0.78539816, 0.   , -1.57079633]
# [ 1.57079633,  1.15889862, 0.254,  1.57079633]


class Ecm(Arm):
    NAME = 'ECM'
    URDF_PATH = os.path.join(ASSET_DIR_PATH, 'ecm/ecm.urdf')
    DoF = 4  # 4-dof arm
    JOINT_TYPES = ('R', 'R', 'P', 'R')
    EEF_LINK_INDEX = 4   # EEF link index, one redundant joint for inverse kinematics
    TIP_LINK_INDEX = 5   # redundant joint for easier camera matrix computation
    RCM_LINK_INDEX = 10  # RCM link index
    # D-H parameters
    A     = np.array([0.0, 0.0, 0.0, 0.0])
    ALPHA = np.array([np.pi / 2, -np.pi / 2, np.pi / 2, 0.0])
    D     = np.array([0.0, 0.0, -0.3822, 0.3829])
    THETA = np.array([np.pi / 2, -np.pi / 2, 0.0, 0.0])

    def __init__(self, pos=(0., 0., 1.), orn=(0., 0., 0., 1.),
                 scaling=1.):
        super(Ecm, self).__init__(self.URDF_PATH, pos, orn,
                                  TOOL_JOINT_LIMIT, tool_T_tip, scaling)

        # camera control related parameters
        self.view_matrix = None
        self.proj_matrix = None
        self._homo_delta = np.zeros((2, 1))
        self._wz = 0

        # b: rcm, e: eef, c: camera
        pos_eef, orn_eef = get_link_pose(self.body, self.EEF_LINK_INDEX)
        pos_cam, orn_cam = get_link_pose(self.body, self.TIP_LINK_INDEX)
        self._tip_offset = np.linalg.norm(np.array(pos_eef) - np.array(pos_cam))  # TODO
        wRe = np.array(p.getMatrixFromQuaternion(orn_eef)).reshape((3, 3))
        wRc = np.array(p.getMatrixFromQuaternion(orn_cam)).reshape((3, 3))
        self._wRc0 = wRc.copy()  # initial rotation matrix
        self._eRc = np.matmul(wRe.T, wRc)

    def _get_joint_positions_all(self, abs_input):
        """ With the consideration of parallel mechanism constraints and other redundant joints.
        """
        positions = get_joint_positions(self.body, self.joints)
        joint_positions = [
            abs_input[0], abs_input[1],  # 0, 1
            abs_input[2] * self.scaling, abs_input[3],  # 2, 3
            positions[4], positions[5],  # 4 (0.0), 5 (0.0)
            abs_input[1],  # 6
            -abs_input[1], -abs_input[1],  # 7, 8
            abs_input[1],  # 9
            positions[10],  # 10 (0.0)
        ]
        return joint_positions

    def cVc_to_dq(self, cVc: np.ndarray) -> np.ndarray:
        """
        convert the camera velocity in its own frame (cVc) into the joint velocity q_dot
        """
        cVc = cVc.reshape((3, 1))

        # restrict the step size, need tune
        if np.abs(cVc).max() > 0.01:
            cVc = cVc / np.abs(cVc).max() * 0.01

        # Forward kinematics
        q = self.get_current_joint_position()
        bRe = self.robot.fkine(q).R  # use rtb instead of PyBullet, no tool_tip_offset
        _, orn_cam = get_link_pose(self.body, self.TIP_LINK_INDEX)
        wRc = np.array(p.getMatrixFromQuaternion(orn_cam)).reshape((3, 3))

        # Rotation
        R1, R2 = self._wRc0, wRc
        x = R1[0, 0] * R2[1, 0] - R1[1, 0] * R2[0, 0] + R1[0, 1] * R2[1, 1] - R1[1, 1] * R2[0, 1]
        y = R1[0, 0] * R2[1, 1] - R1[1, 0] * R2[0, 1] - R1[0, 1] * R2[1, 0] + R1[1, 1] * R2[0, 0]
        dz = np.arctan(x / y)
        k1, k2 = 25.0, 0.1
        self._wz = k1 * dz * np.exp(-k2 * np.linalg.norm(self._homo_delta))
        # print(' -> x: {:.4f}, y: {:.4f}, dz: {:.4f}, wz: {:.4f}'.format(x, y, dz, self._wz))

        # Pseudo Solution
        d = self._tip_offset
        Jd = np.matmul(self._eRc,
                       np.array([[0,  0, d, 0],
                                 [0, -d, 0, 0],
                                 [1,  0, 0, 0]]))
        Je = np.matmul(self._eRc,
                       np.array([[0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))

        eVe4 = np.dot(np.linalg.pinv(Jd), cVc) \
               + np.dot(np.dot((np.eye(4) - np.dot(np.linalg.pinv(Jd), Jd)), np.linalg.pinv(Je)),
                        np.array([[0], [0], [self._wz]]))
        eVe = np.zeros((6, 1))
        eVe[2: 6] = eVe4[0: 4]
        Q = np.zeros((6, 6))
        Q[0: 3, 0: 3] = - bRe
        Q[3: 6, 3: 6] = - bRe
        bVe = np.dot(Q, eVe)

        # Compute the Jacobian matrix
        bJe = self.get_jacobian_spatial()
        dq = np.dot(np.linalg.pinv(bJe), bVe)
        # print(" -> cVc: {}, q: {}, dq: {}".format(list(np.round(cVc.flatten(), 4)), q, list(dq.flatten())))
        return dq.flatten()

    def render_image(self, width=RENDER_WIDTH, height=RENDER_HEIGHT):
        pos_eef, orn_eef = get_link_pose(self.body, self.EEF_LINK_INDEX)
        pos_tip = get_link_pose(self.body, self.TIP_LINK_INDEX)[0]
        mat_eef = np.array(p.getMatrixFromQuaternion(orn_eef)).reshape((3, 3))

        # TODO: need to check the up vector
        self.view_matrix = p.computeViewMatrix(cameraEyePosition=pos_eef,
                                               cameraTargetPosition=pos_tip,
                                               cameraUpVector=mat_eef[:, 0])
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=FoV,
                                                        aspect=float(width) / height,
                                                        nearVal=0.01,
                                                        farVal=10.0)

        rgb_array, mask = render_image(width, height,
                                       self.view_matrix, self.proj_matrix)
        return rgb_array, mask

    def get_centroid_proj(self, pos) -> np.ndarray:
        """
        Compute the object position in the camera NDC space.
        Refer to OpenGL.
        :param pos: object position in the world frame.
        """
        assert len(pos) in (3, 4)
        if len(pos) == 3:
            # homogeneous coordinates: (x, y, z) -> (x, y, z, w)
            pos_obj = np.ones((4, 1))
            pos_obj[: 3, 0] = pos
        else:
            pos_obj = np.array(pos).reshape((4, 1))

        view_matrix = np.array(self.view_matrix).reshape(4, 4).T
        proj_matrix = np.array(self.proj_matrix).reshape(4, 4).T
        # pos in the camera frame
        pos_cam = np.dot(proj_matrix, np.dot(view_matrix, pos_obj))
        pos_cam /= pos_cam[3, 0]
        return np.array([pos_cam[0][0], - pos_cam[1][0]])  # be consistent with get_centroid

    @property
    def homo_delta(self):
        return self._homo_delta

    @homo_delta.setter
    def homo_delta(self, val: np.ndarray):
        self._homo_delta = val

    @property
    def wz(self):
        return self._wz
