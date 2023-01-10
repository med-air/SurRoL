import os
from re import S
import time
import numpy as np

import pybullet as p
from surrol.tasks.ecm_env import EcmEnv, goal_distance
from surrol.utils.pybullet_utils import (
    get_body_pose,
)
from surrol.utils.utils import RGB_COLOR_255, Boundary, get_centroid
from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
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

class StaticTrack(EcmEnv):
    ACTION_MODE = 'cVc'
    DISTANCE_THRESHOLD = 0.02
    QPOS_ECM = (0, 0, 0.04, 0)
    WORKSPACE_LIMITS = ((-0.5, 0.5), (-0.4, 0.4), (0.05, 0.05))
    CUBE_NUMBER = 18
    QPOS_ECM = (0, 0.6, 0.04, 0)
    POSE_TABLE = ((0.5, 0, 0.001), (0, 0, 0))

    ACTION_ECM_SIZE=3
    def __init__(self, render_mode=None, cid = -1):
        super(StaticTrack, self).__init__(render_mode, cid)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0.27, -0.2, 0.55),
            distance=2.3,
            yaw=150,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
    def _env_setup(self):
        super(StaticTrack, self)._env_setup()
        self.use_camera = True
        self.misorientation_threshold = 0.01

        # robot
        self.ecm.reset_joint(self.QPOS_ECM)

        # target cube
        b = Boundary(self.workspace_limits)
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'cube/cube.urdf'),
                            (0, 0, 0.05),
                            p.getQuaternionFromEuler(np.random.uniform(np.deg2rad([0, 0, -90]),
                                                                       np.deg2rad([0, 0, 90]))),
                            globalScaling=self.SCALING)
        color = RGB_COLOR_255[0]
        p.changeVisualShape(obj_id, -1,
                            rgbaColor=(color[0] / 255, color[1] / 255, color[2] / 255, 1),
                            specularColor=(0., 0., 0.))
        self.obj_ids['fixed'].append(obj_id)  # 0 (target)
        self.obj_id = obj_id
        b.add(obj_id, min_distance=0.15)

        # other cubes
        for i in range(self.CUBE_NUMBER):
            obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'cube/cube.urdf'),
                                (0, 0, 0.05), (0, 0, 0, 1),
                                globalScaling=self.SCALING)
            color = RGB_COLOR_255[1 + i // 2]
            p.changeVisualShape(obj_id, -1,
                                rgbaColor=(color[0] / 255, color[1] / 255, color[2] / 255, 1),
                                specularColor=(0., 0., 0.))
            b.add(obj_id, min_distance=0.15)

    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state()

        _, mask = self.ecm.render_image()
        in_view, centroids = get_centroid(mask, self.obj_id)

        if not in_view:
            # out of view; differ when the object is on the boundary.
            pos, _ = get_body_pose(self.obj_id)
            centroids = self.ecm.get_centroid_proj(pos)
            print(" -> Out of view! {}".format(np.round(centroids, 4)))

        achieved_goal = np.array([
            centroids[0], centroids[1], self.ecm.wz
        ])

        observation = np.concatenate([
            robot_state, np.array(in_view).astype(np.float).ravel(),
            centroids.ravel(), np.array(self.ecm.wz).ravel()  # achieved_goal.copy(),
        ])
        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
        return obs

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = goal_distance(achieved_goal[..., :2], desired_goal[..., :2])
        misori = np.abs(achieved_goal[..., 2] - achieved_goal[..., 2])
        print(f"ECM static track: {d} {self.distance_threshold} {d < self.distance_threshold} {misori} {misori < self.misorientation_threshold}")
        print(np.logical_and(
            d < self.distance_threshold,
            misori < self.misorientation_threshold
        ).astype(np.float32))

        return np.logical_and(
            d < self.distance_threshold,
            misori < self.misorientation_threshold
        ).astype(np.float32)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array([0., 0., 0.])
        return goal.copy()

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        cam_u = obs['achieved_goal'][0] * RENDER_WIDTH
        cam_v = obs['achieved_goal'][1] * RENDER_HEIGHT
        self.ecm.homo_delta = np.array([cam_u, cam_v]).reshape((2, 1))
        if np.linalg.norm(self.ecm.homo_delta) < 1 and np.linalg.norm(self.ecm.wz) < 0.1:
            # e difference is small enough
            action = np.zeros(3)
        else:
            print("Pixel error: {:.4f}".format(np.linalg.norm(self.ecm.homo_delta)))
            # controller
            fov = np.deg2rad(FoV)
            fx = (RENDER_WIDTH / 2) / np.tan(fov / 2)
            fy = (RENDER_HEIGHT / 2) / np.tan(fov / 2)  # TODO: not sure
            cz = 1.0
            Lmatrix = np.array([[-fx / cz, 0., cam_u / cz],
                                [0., -fy / cz, cam_v / cz]])
            action = 0.5 * np.dot(np.linalg.pinv(Lmatrix), self.ecm.homo_delta).flatten() / 0.01
            if np.abs(action).max() > 1:
                action /= np.abs(action).max()
            action *= 0.8
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
    env = StaticTrack(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
