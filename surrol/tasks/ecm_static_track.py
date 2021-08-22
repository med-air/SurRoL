import os
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


class StaticTrack(EcmEnv):
    ACTION_MODE = 'cVc'
    DISTANCE_THRESHOLD = 0.01
    QPOS_ECM = (0, 0, 0.04, 0)
    WORKSPACE_LIMITS = ((-0.5, 0.5), (-0.4, 0.4), (0.05, 0.05))
    CUBE_NUMBER = 18

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


if __name__ == "__main__":
    env = StaticTrack(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
