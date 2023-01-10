import os
import time
import numpy as np

import pybullet as p
from surrol.const import ASSET_DIR_PATH
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera,    
    wrap_angle
)
from surrol.tasks.ecm_env import EcmEnv, goal_distance

from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm

class ECMReach(EcmEnv):
    ACTION_MODE = 'dmove'
    WORKSPACE_LIMITS = ((0.45, 0.55), (-0.05, 0.05), (0.60, 0.70))
    QPOS_ECM = (0, 0.6, 0.04, 0)
    POSE_TABLE = ((0.5, 0, 0.001), (0, 0, 0))

    ACTION_ECM_SIZE=3
    def __init__(self, render_mode=None, cid = -1):
        super(ECMReach, self).__init__(render_mode, cid)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0.27, -0.2, 0.55),
            distance=2.2,
            yaw=150,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
    def _env_setup(self):
        super(ECMReach, self)._env_setup()

        # robot
        pos = ((self.workspace_limits[0][1] + self.workspace_limits[0][0]) / 2,
               (self.workspace_limits[1][1] + self.workspace_limits[1][0]) / 2,
               self.workspace_limits[2][1])
        joint_positions = self.ecm.inverse_kinematics((pos, None), self.ecm.EEF_LINK_INDEX)
        self.ecm.reset_joint(joint_positions[:self.ecm.DoF])

        # for goal plotting
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'sphere/sphere.urdf'),
                            globalScaling=self.SCALING * 3)
        self.obj_ids['fixed'].append(obj_id)  # 0

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.random.rand(3) * (self.workspace_limits[:, 1] - self.workspace_limits[:, 0]) \
               + self.workspace_limits[:, 0]
        return goal.copy()

    def _sample_goal_callback(self):
        """ Set the target pose for visualization
        """
        p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], self.goal, (0, 0, 0, 1))

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        pos_d, _ = self.ecm.pose_world2rcm((obs['desired_goal'], None))
        pos_a, _ = self.ecm.pose_world2rcm((obs['achieved_goal'], None))
        delta_pos = (np.array(pos_d) - np.array(pos_a)) / 0.01
        if np.abs(delta_pos).max() > 1:
            delta_pos /= np.abs(delta_pos).max()
        delta_pos *= 0.2
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2]])
        return action


    def _set_action_ecm(self, action):
        action *= 0.01 * self.SCALING
        pose_rcm = self.ecm.get_current_position()
        pose_rcm[:3, 3] += action
        pos, _ = self.ecm.pose_rcm2world(pose_rcm, 'tuple')
        joint_positions = self.ecm.inverse_kinematics((pos, None), self.ecm.EEF_LINK_INDEX)  # do not consider orn
        self.ecm.move_joint(joint_positions[:self.ecm.DoF])
    def _reset_ecm_pos(self):
        self.ecm.reset_joint(self.QPOS_ECM)

if __name__ == "__main__":
    env = ECMReach(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
