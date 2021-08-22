import os
import time

import pybullet as p
from surrol.tasks.ecm_env import EcmEnv, goal_distance
from surrol.utils.pybullet_utils import (
    get_body_pose,
)
from surrol.utils.utils import RGB_COLOR_255, Boundary, Trajectory, get_centroid
from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
import numpy as np


class ActiveTrack(EcmEnv):
    """
    Active track is not a GoalEnv since the environment changes internally.
    The reward is shaped.
    """
    ACTION_MODE = 'cVc'
    QPOS_ECM = (0, 0, 0.02, 0)
    WORKSPACE_LIMITS = ((-0.3, 0.6), (-0.4, 0.4), (0.05, 0.05))
    CUBE_NUMBER = 18

    def __init__(self, render_mode=None):
        # to control the step
        self._step = 0

        super(ActiveTrack, self).__init__(render_mode)

    def step(self, action: np.ndarray):
        obs, reward, done, info = super().step(action)
        centroid = obs[-3: -1]
        if not (-1.2 < centroid[0] < 1.1 and -1.1 < centroid[1] < 1.1):
            # early stop if out of view
            done = True
        info['achieved_goal'] = centroid
        return obs, reward, done, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """ Dense reward."""
        centroid, wz = achieved_goal[-3: -1], achieved_goal[-1]
        d = goal_distance(centroid, desired_goal) / 2
        reward = 1 - (d + np.linalg.norm(wz) * 0.1)  # maximum reward is 1, important for baseline DDPG
        return reward

    def _env_setup(self):
        super(ActiveTrack, self)._env_setup()
        self.use_camera = True

        # robot
        self.ecm.reset_joint(self.QPOS_ECM)

        # trajectory
        traj = Trajectory(self.workspace_limits, seed=None)
        self.traj = traj
        self.traj.set_step(self._step)

        # target cube
        b = Boundary(self.workspace_limits)
        x, y = self.traj.step()
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'cube/cube.urdf'),
                            (x * self.SCALING, y * self.SCALING, 0.05 * self.SCALING),
                            p.getQuaternionFromEuler(np.random.uniform(np.deg2rad([0, 0, -90]),
                                                                       np.deg2rad([0, 0, 90]))),
                            globalScaling=0.8 * self.SCALING)
        color = RGB_COLOR_255[0]
        p.changeVisualShape(obj_id, -1,
                            rgbaColor=(color[0] / 255, color[1] / 255, color[2] / 255, 1),
                            specularColor=(0.1, 0.1, 0.1))
        self.obj_ids['fixed'].append(obj_id)  # 0 (target)
        self.obj_id = obj_id
        b.add(obj_id, sample=False, min_distance=0.12)
        self._cid = p.createConstraint(obj_id, -1, -1, -1,
                                       p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [x, y, 0.05 * self.SCALING])

        # other cubes
        b.set_boundary(self.workspace_limits + np.array([[-0.2, 0], [0, 0], [0, 0]]))
        for i in range(self.CUBE_NUMBER):
            obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'cube/cube.urdf'),
                                (0, 0, 0.05), (0, 0, 0, 1),
                                globalScaling=0.8 * self.SCALING)
            color = RGB_COLOR_255[1 + i // 2]
            p.changeVisualShape(obj_id, -1,
                                rgbaColor=(color[0] / 255, color[1] / 255, color[2] / 255, 1),
                                specularColor=(0.1, 0.1, 0.1))
            # p.changeDynamics(obj_id, -1, mass=0.01)
            b.add(obj_id, min_distance=0.12)

    def _get_obs(self) -> np.ndarray:
        robot_state = self._get_robot_state()
        # may need to modify
        _, mask = self.ecm.render_image()
        in_view, centroids = get_centroid(mask, self.obj_id)

        if not in_view:
            # out of view; differ when the object is on the boundary.
            pos, _ = get_body_pose(self.obj_id)
            centroids = self.ecm.get_centroid_proj(pos)
            print(" -> Out of view! {}".format(np.round(centroids, 4)))

        observation = np.concatenate([
            robot_state, np.array(in_view).astype(np.float).ravel(),
            centroids.ravel(), np.array(self.ecm.wz).ravel()  # achieved_goal.copy(),
        ])
        return observation

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array([0., 0.])
        return goal.copy()

    def _step_callback(self):
        """ Move the target along the trajectory
        """
        for _ in range(10):
            x, y = self.traj.step()
            self._step = self.traj.get_step()
            pivot = [x, y, 0.05 * self.SCALING]
            p.changeConstraint(self._cid, pivot, maxForce=50)
            p.stepSimulation()

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        centroid = obs[-3: -1]
        cam_u = centroid[0] * RENDER_WIDTH
        cam_v = centroid[1] * RENDER_HEIGHT
        self.ecm.homo_delta = np.array([cam_u, cam_v]).reshape((2, 1))
        if np.linalg.norm(self.ecm.homo_delta) < 8 and np.linalg.norm(self.ecm.wz) < 0.1:
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
        return action


if __name__ == "__main__":
    env = ActiveTrack(render_mode='human')  # create one process and corresponding env

    env.test(horizon=200)
    env.close()
    time.sleep(2)
