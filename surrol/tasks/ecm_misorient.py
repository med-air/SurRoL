import os
import time
import numpy as np
import cv2

import pybullet as p
from surrol.tasks.ecm_env import EcmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
)
from surrol.utils.robotics import (
    get_matrix_from_pose_2d,
    get_intrinsic_matrix,
)
from surrol.utils.utils import RGB_COLOR_255
from surrol.const import ASSET_DIR_PATH


class MisOrient(EcmEnv):
    ACTION_SIZE = 1  # droll
    ACTION_MODE = 'droll'
    DISTANCE_THRESHOLD = 0.01
    QPOS_ECM = (0, 0, 0.05, 0)
    WORKSPACE_LIMITS = ((-0.6, 0.4), (-0.8, 0.8), (0.02, 0.02))
    CUBE_NUMBER = (9, 16)

    def _env_setup(self):
        super(MisOrient, self)._env_setup()
        self.use_camera = True

        # Natural of light transformation
        pos, orn = get_link_pose(self.ecm.body, self.ecm.TIP_LINK_INDEX)
        self.cam_nof = get_matrix_from_pose_2d((pos, orn))
        self.cam_nof[:3, 3] /= self.SCALING

        # robot
        low = np.array([np.deg2rad(-90.0) / 2, np.deg2rad(-45.0) / 2, 0., np.deg2rad(-90.0) / 2])
        high = np.array([np.deg2rad(90.0) / 2, np.deg2rad(66.4) / 2, 0.24, np.deg2rad(90.0) / 2])
        joint_positions = np.random.uniform(low, high)
        self.ecm.reset_joint(joint_positions)

        # # cube; disable to speed up training
        # cube_num = self.CUBE_NUMBER
        # x = np.linspace(self.workspace_limits[0, 0], self.workspace_limits[0, 1], cube_num[0])
        # y = np.linspace(self.workspace_limits[1, 0], self.workspace_limits[1, 1], cube_num[1])
        # textures = os.listdir(os.path.join(ASSET_DIR_PATH, 'cube/texture'))
        # textures = [tex for tex in textures if tex.startswith('cube_tex') and tex.endswith('.png')]
        # for i in range(cube_num[0]):
        #     for j in range(cube_num[1]):
        #         # little use during training
        #         obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'cube/cube_vis.urdf'),
        #                             (x[i], y[j], 0.04), (0, 0, 0, 1),
        #                             useFixedBase=True, globalScaling=0.4)
        #         if np.random.rand() > 0.6:
        #             color = RGB_COLOR_255[np.random.randint(len(RGB_COLOR_255))]
        #         else:
        #             color = [225, 225, 225]
        #         p.changeVisualShape(obj_id, -1,
        #                             rgbaColor=(color[0] / 255, color[1] / 255, color[2] / 255, 1),
        #                             specularColor=(0.1, 0.1, 0.1))
        #         tex = textures[np.random.randint(len(textures))]
        #         tex_id = p.loadTexture(os.path.join(ASSET_DIR_PATH, 'cube/texture', tex))
        #         p.changeVisualShape(obj_id, -1, textureUniqueId=tex_id)

    @staticmethod
    def _get_misorientation(tip_world: np.ndarray, tip_world0: np.ndarray) -> np.ndarray:
        """ Get the misorientation angle """
        # Natural of light projection model
        nof_w, nof_h = 1920, 1080
        nof_c = get_intrinsic_matrix(nof_w, nof_h, np.deg2rad(120))
        # Current vision sensor projection model
        cam_w, cam_h = 512, 512
        cam_u, cam_v = cam_w / 2.0, cam_h / 2.0
        cam_c = get_intrinsic_matrix(cam_w, cam_h, np.deg2rad(60))

        # calculate the intersect point of camera axis with the plane z=z0, for current vision sensor
        cam_z = tip_world[0: 3, 2]
        cam_pos = tip_world[0: 3, 3]
        plane_z = np.array([0., 0., 1.])
        z0 = 0
        plane_pos = np.array([0, 0, z0])
        t = ((plane_pos[0] - cam_pos[0]) * plane_z[0] +
             (plane_pos[1] - cam_pos[1]) * plane_z[1] +
             (plane_pos[2] - cam_pos[2]) * plane_z[2]) \
            / (plane_z[0] * cam_z[0] + plane_z[1] * cam_z[1] + plane_z[2] * cam_z[2])  # intersection time
        inter_px = cam_pos[0] + t * cam_z[0]
        inter_py = cam_pos[1] + t * cam_z[1]
        inter_pz = cam_pos[2] + t * cam_z[2]

        tip_world_inv = np.linalg.inv(tip_world)
        cam_P = np.dot(cam_c, tip_world_inv[0:3, 0:4])
        # points in the world coordinates, o-xyz coordinates
        pts = np.array([[inter_px, inter_py, inter_pz, 1],
                        [inter_px + 0.1, inter_py, inter_pz, 1],
                        [inter_px, inter_py + 0.1, inter_pz, 1],
                        [inter_px, inter_py, inter_pz + 0.05, 1.0]]).T
        P_xy_axis = np.dot(cam_P, pts)
        origin = P_xy_axis[0:2, 0] / P_xy_axis[2, 0]
        true_origin = np.array([cam_w - origin[0], cam_h - origin[1]])  # need to be verify in python

        dx = P_xy_axis[0: 2, 1] / P_xy_axis[2, 1]
        true_dx = np.array([cam_w - dx[0], cam_h - dx[1]])
        dy = P_xy_axis[0: 2, 2] / P_xy_axis[2, 2]
        true_dy = np.array([cam_w - dy[0], cam_h - dy[1]])
        dz = P_xy_axis[0: 2, 3] / P_xy_axis[2, 3]
        true_dz = np.array([cam_w - dz[0], cam_h - dz[1]])
        x_axis = true_dx - true_origin
        y_axis = true_dy - true_origin
        z_axis = true_dz - true_origin
        # points in the current image
        pts1 = np.array([[cam_u, cam_v],
                         [cam_u + float(x_axis[0]), cam_v + float(x_axis[1])],
                         [cam_u + float(y_axis[0]), cam_v + float(y_axis[1])]]).astype(np.float32)

        # points in the target image, need to keep the center the same, only calculate the direction
        tip_world0_inv = np.linalg.inv(tip_world0)
        nof_P = np.dot(nof_c, tip_world0_inv[0:3, 0:4])
        P_xy_axis = np.dot(nof_P, pts)
        origin = P_xy_axis[0: 2, 0] / P_xy_axis[2, 0]
        true_origin = np.array([nof_w - origin[0], nof_h - origin[1]])

        dx = P_xy_axis[0: 2, 1] / P_xy_axis[2, 1]
        true_dx = np.array([nof_w - dx[0], nof_h - dx[1]])
        dy = P_xy_axis[0: 2, 2] / P_xy_axis[2, 2]
        true_dy = np.array([nof_w - dy[0], nof_h - dy[1]])
        dz = P_xy_axis[0: 2, 3] / P_xy_axis[2, 3]
        true_dz = np.array([nof_w - dz[0], nof_h - dz[1]])
        current_x_axis = true_dx - true_origin
        current_y_axis = true_dy - true_origin
        current_z_axis = true_dz - true_origin
        # points in the NOF image, only keep the direction as the same
        pts2 = np.array([[cam_u, cam_v],
                         [cam_u + float(current_x_axis[0]), cam_v + float(current_x_axis[1])],
                         [cam_u + float(current_y_axis[0]), cam_v + float(current_y_axis[1])]]).astype(np.float32)

        M = cv2.getAffineTransform(pts1, pts2)
        # SVD decomposition
        [U, S, V] = np.linalg.svd(M[0: 2, 0: 2])
        R_theta = np.dot(U, V.T)
        R_phi = np.dot(V, np.dot(S, V.T))
        ratio = S[0] / S[1]
        theta = np.arccos(R_theta[0, 0])  # need to return: only return the abs value
        theta_symbol = np.arcsin(R_theta[1, 0])  # return with direction
        phi = np.arccos(V[0, 0])
        return np.array([-theta_symbol])

    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state()

        if self._render_mode == 'human':
            _, mask = self.ecm.render_image()

        pos, orn = get_link_pose(self.ecm.body, self.ecm.TIP_LINK_INDEX)
        cam_world = get_matrix_from_pose_2d((pos, orn))
        cam_world[:3, 3] /= self.SCALING

        mis_orient = self._get_misorientation(cam_world, self.cam_nof)

        achieved_goal = mis_orient.copy()

        observation = np.concatenate([
            robot_state, np.array(p.getEulerFromQuaternion(orn))  # achieved_goal.copy(),
        ])
        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
        return obs

    def _sample_goal(self) -> np.ndarray:
        """Samples a new goal and returns it.
        """
        goal = np.array([0.])
        return goal.copy()

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        pos, orn = get_link_pose(self.ecm.body, self.ecm.TIP_LINK_INDEX)
        cam_world = get_matrix_from_pose_2d((pos, orn))
        cam_world[:3, 3] /= self.SCALING
        droll = self._get_misorientation(cam_world, self.cam_nof) / np.deg2rad(3)
        action = droll.clip(-1, 1) * 0.3
        return action


if __name__ == "__main__":
    env = MisOrient(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
