import numpy as np
import pybullet as p
from surrol.gym.surrol_env import SurRoLEnv, RENDER_HEIGHT
from surrol.robots.ecm import Ecm
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera
)
from surrol.utils.robotics import get_pose_2d_from_matrix


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class EcmEnv(SurRoLEnv):
    """
    Single arm env using ECM (active_track is not a GoalEnv)
    Refer to Gym fetch
    https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch_env.py
    ravens
    https://github.com/google-research/ravens/blob/master/ravens/environments/environment.py
    """
    ACTION_SIZE = 3  # (dx, dy, dz) or cVc or droll (1)
    ACTION_MODE = 'cVc'
    DISTANCE_THRESHOLD = 0.005
    POSE_ECM = ((-0.05, 0, 1.0024), (0, 30 / 180 * np.pi, 0))
    QPOS_ECM = (0, 0, 0.05, 0)
    WORKSPACE_LIMITS = ((0.45, 0.55), (-0.05, 0.05), (0.60, 0.70))
    SCALING = 1.

    def __init__(self, render_mode: str = None):
        # workspace
        self.workspace_limits = np.asarray(self.WORKSPACE_LIMITS)
        self.workspace_limits *= self.SCALING

        # camera
        self.use_camera = False

        # has_object
        self.has_object = False
        self.obj_id = None

        super(EcmEnv, self).__init__(render_mode)

        # change duration
        self._duration = 0.1

        # distance_threshold
        self.distance_threshold = self.DISTANCE_THRESHOLD * self.SCALING

        # render related setting
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0.27 * self.SCALING, -0.20 * self.SCALING, 0.55 * self.SCALING),
            distance=1.80 * self.SCALING,
            yaw=150,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )

    def render(self, mode='rgb_array'):
        # TODO: check how to disable specular color when using EGL renderer
        if mode != "rgb_array":
            return np.array([])
        rgb_array = super().render(mode)
        if self.use_camera:
            rgb_cam, _ = self.ecm.render_image(RENDER_HEIGHT, RENDER_HEIGHT)
            rgb_array = np.concatenate((rgb_array, rgb_cam), axis=1)
        return rgb_array

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """ Sparse reward. """
        # d = goal_distance(achieved_goal, desired_goal)
        # return - (d > self.distance_threshold).astype(np.float32)
        return self._is_success(achieved_goal, desired_goal).astype(np.float32) - 1.

    def _env_setup(self):
        assert self.ACTION_MODE in ('cVc', 'dmove', 'droll')
        # camera
        if self._render_mode == 'human':
            reset_camera(yaw=150.0, pitch=-30.0, dist=1.50 * self.SCALING,
                         target=(0.27 * self.SCALING, -0.20 * self.SCALING, 0.55 * self.SCALING))

        # robot
        self.ecm = Ecm(self.POSE_ECM[0], p.getQuaternionFromEuler(self.POSE_ECM[1]),
                       scaling=self.SCALING)

        pass  # need to implement based on every task
        # self.obj_ids

    def _get_robot_state(self) -> np.ndarray:
        # TODO
        # robot state: eef pose in the ECM coordinate
        pose_rcm = get_pose_2d_from_matrix(self.ecm.get_current_position())
        return np.concatenate([
            np.array(pose_rcm[0]), np.array(p.getEulerFromQuaternion(pose_rcm[1]))
        ])

    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state()
        # may need to modify
        if self.has_object:
            pos, _ = get_link_pose(self.obj_id, -1)
            object_pos = np.array(pos)
        else:
            object_pos = np.zeros(0)

        if self.has_object:
            achieved_goal = object_pos.copy()
        else:
            achieved_goal = np.array(get_link_pose(self.ecm.body, self.ecm.EEF_LINK_INDEX)[0])  # eef position

        observation = np.concatenate([
            robot_state, object_pos.ravel(),  # achieved_goal.copy(),
        ])
        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
        return obs

    def _set_action(self, action: np.ndarray):
        """
        delta_position (3) and delta_theta (1); in world coordinate
        """
        assert len(action) == self.ACTION_SIZE
        action = action.copy()  # ensure that we don't change the action outside of this scope
        if self.ACTION_MODE == 'cVc':
            # hyper-parameters are sensitive; need to tune
            # if np.linalg.norm(action) > 1:
            #     action /= np.linalg.norm(action)
            action *= 0.01 * self.SCALING  # velocity (HeadPitch, HeadYaw), limit maximum change in velocity
            dq = 0.05 * self.ecm.cVc_to_dq(action)  # scaled
            self.ecm.dmove_joint(dq)
        elif self.ACTION_MODE == 'dmove':
            # Incremental motion in cartesian space in the base frame
            action *= 0.01 * self.SCALING
            pose_rcm = self.ecm.get_current_position()
            pose_rcm[:3, 3] += action
            pos, _ = self.ecm.pose_rcm2world(pose_rcm, 'tuple')
            joint_positions = self.ecm.inverse_kinematics((pos, None), self.ecm.EEF_LINK_INDEX)  # do not consider orn
            self.ecm.move_joint(joint_positions[:self.ecm.DoF])
        elif self.ACTION_MODE == 'droll':
            # change the roll angle
            action *= np.deg2rad(3)
            self.ecm.dmove_joint_one(action[0], 3)
        else:
            raise NotImplementedError

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        raise NotImplementedError

    @property
    def action_size(self):
        return self.ACTION_SIZE

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a scripted oracle strategy
        """
        raise NotImplementedError
