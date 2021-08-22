"""
Helper functions for robotics-related calculation
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_matrix_from_pose_2d(pose: tuple) -> np.ndarray:
    """
    :param pose: tuple (position (3), orientation quaternion (4) or euler (3, xyz))
    :return: matrix (4*4)
    """
    mat = np.eye(4)
    if pose[1] is not None:
        assert len(pose[1]) in (3, 4)
        if len(pose[1]) == 4:
            # quaternion
            mat[:3, :3] = R.from_quat(pose[1]).as_matrix()
        else:
            # xyz euler
            mat[:3, :3] = R.from_euler('xyz', pose[1]).as_matrix()
    else:
        # for dummy orientation
        mat[:3, :3] = R.from_quat((0, 0, 0, 1)).as_matrix()
    mat[:3, 3] = pose[0]
    return mat


def get_pose_2d_from_matrix(mat: np.ndarray) -> tuple:
    """
    :param mat: matrix (4*4)
    :return: (position (3), orientation (4))
    """
    pose = (tuple(mat[:3, 3]), tuple(R.from_matrix(mat[:3, :3]).as_quat()))
    return pose


def get_euler_from_matrix(mat):
    """
    :param mat: rotation matrix (3*3)
    :return: rotation in 'xyz' euler
    """
    rot = R.from_matrix(mat)
    return rot.as_euler('xyz', degrees=False)


def get_matrix_from_euler(ori):
    """
    :param ori: rotation in 'xyz' euler
    :return: rotation matrix (3*3)
    """
    rot = R.from_euler('xyz', ori)
    return rot.as_matrix()


def get_intrinsic_matrix(width, height, fov):
    """ Calculate the camera intrinsic matrix.
    """
    fy = fx = (width / 2.) / np.tan(fov / 2.)  # fy = fy?
    cx, cy = width / 2., height / 2.
    mat = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]]).astype(np.float)
    return mat
