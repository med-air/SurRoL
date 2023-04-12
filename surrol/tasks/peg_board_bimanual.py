import os
import time
import numpy as np
import random
import pybullet as p
import pybullet_data

from surrol.tasks.psm_env import PsmsEnv, goal_distance
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera,
    wrap_angle
)
from surrol.utils.robotics import (
    get_euler_from_matrix,
    get_matrix_from_euler
)
from surrol.const import ASSET_DIR_PATH
from surrol.tasks.ecm_env import EcmEnv, goal_distance

from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.robots.ecm import Ecm

class BiPegBoard(PsmsEnv):
    POSE_BOARD = ((0.55, 0, 0.6861), (0, 0, 0))  # 0.675 + 0.011 + 0.001
    WORKSPACE_LIMITS1 = ((0.47, 0.66), (-0., 0.10), (0.606, 0.785))
    WORKSPACE_LIMITS2 = ((0.47, 0.66), (-0.15, 0.), (0.606, 0.785))
    SCALING = 5.
    DISTANCE_THRESHOLD = 0.01
    POSE_PSM1 = ((0.05, 0.25, 0.8224), (0, 0, -(90 + 20) / 180 * np.pi))
    POSE_PSM2 = ((0.05, -0.26, 0.8524), (0, 0, -(90 - 20) / 180 * np.pi))
    QPOS_ECM = (0, 0.75, 0.04, 0)
    ACTION_ECM_SIZE=3

    def _env_setup(self):
        super(BiPegBoard, self)._env_setup()
        self.has_object = True
        # camera
        if self._render_mode == 'human':
            reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
                         target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                       scaling=self.SCALING)
        self.ecm.reset_joint(self.QPOS_ECM)

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1].mean(),
               workspace_limits[2][1])
        orn = p.getQuaternionFromEuler(np.deg2rad([0, 0, -180]))  # reduce difficulty
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        workspace_limits = self.workspace_limits2
        pos = (workspace_limits[0][0],
               workspace_limits[1].mean(),
               workspace_limits[2][1])
        orn = (0.5, 0, 0, -0.5)
        joint_positions = self.psm2.inverse_kinematics((pos, orn), self.psm2.EEF_LINK_INDEX)
        self.psm2.reset_joint(joint_positions)
        self.block_gripper = False

        # board
        asset_scaling = 0.185
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/ring_board.urdf"),
                            np.array(self.POSE_BOARD[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_BOARD[1]),
                            globalScaling=asset_scaling,
                            useFixedBase=1)
        # texture = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/wood.jpg"))
        # p.changeVisualShape(obj_id, -1, textureUniqueId=texture)
        # p.changeVisualShape(obj_id, 0, textureUniqueId=texture)
        self.obj_ids['fixed'].append(obj_id)
        print(f'peg board\' board size: {p.getVisualShapeData(obj_id)}')

        # peg 
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/ring_peg.urdf"),
                            np.array(self.POSE_BOARD[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_BOARD[1]),
                            globalScaling=asset_scaling,
                            useFixedBase=1)
        # texture = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/metal.jpg"))
        # p.changeVisualShape(obj_id, -1, textureUniqueId=texture)
        self._pegs = np.arange(14)
        # for peg in self._pegs:
        #     p.changeVisualShape(obj_id, peg, textureUniqueId=texture)
        np.random.shuffle(self._pegs[:3])
        self.obj_ids['fixed'].append(obj_id)
        # print(f'peg board\' peg size: {p.getVisualShapeData(obj_id)}')

        # rings
        num_rings = 1
        for i in range(num_rings):
            pos, orn = get_link_pose(self.obj_ids['fixed'][-1], i + 2)
            obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, "ring/ring_tao.urdf"),
                                np.array([pos[0] + 0.05 * np.random.random(), pos[1], pos[2]]),
                                p.getQuaternionFromEuler([0, 0, 0]),
                                globalScaling=asset_scaling)
            self.obj_ids['rigid'].append(obj_id)
        self._rings = np.array(self.obj_ids['rigid'][-num_rings:])
        #self._rings = np.array(self.obj_ids['rigid'][-1])
        np.random.shuffle(self._rings)
        for obj_id in self._rings[:1]:
            p.changeVisualShape(obj_id, -1, rgbaColor=(255 / 255, 69 / 255, 58 / 255, 1))
        self.obj_id, self.obj_link1, self.obj_link2 = self._rings[0], 1, 2
        # print(f'peg board\' ring size: {p.getVisualShapeData(obj_id)}')

        # set psm rotation
        self.psm1_eul = np.deg2rad([0, 0, -180])
        self.psm2_eul = np.array([np.deg2rad(-90), 0., np.deg2rad(-180)])

        # render related setting
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.395 * self.SCALING),
            distance=0.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )


    def _set_action(self, action: np.ndarray):
        """
        delta_position (3), delta_theta (1) and open/ close the gripper (1); in world coordinate
        *2 for PSM1 [0: 5] and PSM2 [5: 10]
        """
        assert len(action) == self.ACTION_SIZE
        action = action.copy()  # ensure that we don't change the action outside of this scope
        action[0: 3] *= 0.01 * self.SCALING  # position, limit maximum change in position
        action[5: 8] *= 0.01 * self.SCALING
        for i, psm in enumerate((self.psm1, self.psm2)):
            # set the action for PSM1 and PSM2
            pose_world = psm.pose_rcm2world(psm.get_current_position())
            idx = i * 5
            workspace_limits = self.workspace_limits1 if i == 0 else self.workspace_limits2
            pose_world[:3, 3] = np.clip(pose_world[:3, 3] + action[idx: idx + 3],
                                        workspace_limits[:, 0] - [0.02, 0.02, 0.],
                                        workspace_limits[:, 1] + [0.02, 0.02, 0.08])  # clip to ensure convergence
            rot = get_euler_from_matrix(pose_world[:3, :3])
            psm_eul = self.psm1_eul if i == 0 else self.psm2_eul

            if i == 0:
                action[idx + 3] *= np.deg2rad(15)  # limit maximum change in rotation
                pitch = np.clip(wrap_angle(rot[1] + action[idx + 3]), np.deg2rad(-90), np.deg2rad(90))
                rot = (psm_eul[0], pitch, psm_eul[2]) 

            if i == 1:
                action[idx + 3] *= np.deg2rad(30)  # limit maximum change in rotation
                rot = (psm_eul[0], psm_eul[1], wrap_angle(rot[2] + action[idx + 3]))  # only change yaw

            pose_world[:3, :3] = get_matrix_from_euler(rot)
            action_rcm = psm.pose_world2rcm(pose_world)
            psm.move(action_rcm)

            # jaw
            if self.block_gripper:
                action[idx + 4] = -1
            if action[idx + 4] < 0:
                psm.close_jaw()
                self._activate(i)
            else:
                psm.move_jaw(np.deg2rad(40))
                self._release(i)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        peg_pos = np.array(get_link_pose(self.obj_ids['fixed'][-1], 1)[0])
        goal = np.array([peg_pos[0], peg_pos[1], peg_pos[2] - 0.05])
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = []  # eleven waypoints
        pos_obj1, orn_obj1 = get_link_pose(self.obj_id, self.obj_link1)
        pos_obj2, orn_obj2 = get_link_pose(self.obj_id, self.obj_link2)
        orn1 = p.getEulerFromQuaternion(orn_obj1)
        orn2 = p.getEulerFromQuaternion(orn_obj2)
        orn_eef1 = p.getEulerFromQuaternion(get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1])
        orn_eef2 = p.getEulerFromQuaternion(get_link_pose(self.psm2.body, self.psm2.EEF_LINK_INDEX)[1])
        yaw1 = 0    # horizontal place
        yaw2 = orn2[2] if abs(wrap_angle(orn2[2] - orn_eef2[2])) < abs(wrap_angle(orn2[2] + np.pi - orn_eef2[2])) \
            else wrap_angle(orn2[2] + np.pi)  # minimize the delta yaw

        pos_peg1 = np.array(get_link_pose(self.obj_ids['fixed'][-1], 4)[0])
        pos_peg2 = np.array(get_link_pose(self.obj_ids['fixed'][-1], 5)[0])
        offset_y = 0.0098 * self.SCALING 

        pos_mid1 = [pos_obj1[0] + 0.15, (pos_peg1[1] + pos_peg2[1]) / 2 + offset_y, pos_obj1[2]]  # consider offset
        pos_mid2 = [pos_obj2[0] + 0.15, (pos_peg1[1] + pos_peg2[1]) / 2 - offset_y, pos_obj2[2] + 0.0025 * self.SCALING]  # consider offset

        #-------------------------------Add noise-----------------------------------------
        noise_std = 0.04
        noise = np.clip(noise_std * np.random.random(3), -noise_std, noise_std)
        nsd_pos_mid1 = pos_mid1 + noise
        nsd_pos_mid2 = pos_mid2 + noise

        #----------------------------Subtask 1----------------------------
        self._waypoints.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1], nsd_pos_mid1[2], yaw1, 0.5,
                                         pos_obj2[0], pos_obj2[1], pos_mid2[2], yaw2, 0.5]))  # psm2 above object 0
        self._waypoints.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1], nsd_pos_mid1[2], yaw1, 0.5,
                                         pos_obj2[0], pos_obj2[1], pos_obj2[2], yaw2, 0.5]))  # psm2 pre grasp 1
        self._waypoints.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1], nsd_pos_mid1[2], yaw1, 0.5,
                                         pos_obj2[0], pos_obj2[1], pos_obj2[2], yaw2, -0.5]))  # psm2 grasp 2
        self._waypoints.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1], nsd_pos_mid1[2], yaw1, 0.5,
                                         pos_obj2[0], pos_obj2[1], pos_mid2[2], yaw2, -0.5]))  # psm2 above peg 3
        self._waypoints.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1], nsd_pos_mid1[2], yaw1, 0.5,
                                         nsd_pos_mid2[0], pos_obj2[1], pos_mid2[2], yaw2, -0.5]))  # psm2 move front 4

        #----------------------------Subtask 2----------------------------
        self._waypoints.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1], nsd_pos_mid1[2], yaw1, 0.5,
                                         nsd_pos_mid2[0], nsd_pos_mid2[1], nsd_pos_mid2[2], yaw2, -0.5]))  # psm2 move ro middle 5
        self._waypoints.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1], nsd_pos_mid1[2], yaw1, -0.5,
                                         nsd_pos_mid2[0], nsd_pos_mid2[1], nsd_pos_mid2[2], yaw2, -0.5]))  # psm1 grasp 6 
        self._waypoints.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1], nsd_pos_mid1[2], yaw1, -0.5,
                                         nsd_pos_mid2[0], nsd_pos_mid2[1], nsd_pos_mid2[2], yaw2, 0.5]))   # psm2 release 7
        self._waypoints.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1], nsd_pos_mid1[2], yaw1, -0.5,
                                         nsd_pos_mid2[0], nsd_pos_mid2[1], nsd_pos_mid2[2], yaw2, 0.5]))   # psm2 release 8
        self._waypoints.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1] + 0.02 * self.SCALING, nsd_pos_mid1[2], yaw1, -0.5,
                                         nsd_pos_mid2[0], nsd_pos_mid2[1], nsd_pos_mid2[2], yaw2, 0.5]))   # psm1 move 9

        #----------------------------Subtask 3----------------------------
        target_pos = self.goal.copy()
        yaw1 = np.pi / 2.25
        offset_y = 0.018 * self.SCALING  # offset casued by rotation
        self._waypoints.append(np.array([target_pos[0], target_pos[1] + offset_y, target_pos[2] + 0.03 * self.SCALING, yaw1, -0.5,
                                         nsd_pos_mid2[0], nsd_pos_mid2[1], nsd_pos_mid2[2], yaw2, 0.5]))  # psm1 above target 10
        self._waypoints.append(np.array([target_pos[0], target_pos[1] + offset_y, target_pos[2] + 0.02 * self.SCALING, yaw1, -0.5,
                                         nsd_pos_mid2[0], nsd_pos_mid2[1], nsd_pos_mid2[2], yaw2, 0.5]))  # psm1 above target 11
        self._waypoints.append(np.array([target_pos[0], target_pos[1] + offset_y, target_pos[2] + 0.02 * self.SCALING, yaw1, 0.5,
                                         nsd_pos_mid2[0], nsd_pos_mid2[1], nsd_pos_mid2[2], yaw2, 0.5]))  # psm1 release 12
        self._waypoints_done = [False] * len(self._waypoints)

        #----------------------------Subgoals----------------------------
        self.subgoals = []
        self.subgoals.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1], nsd_pos_mid1[2], nsd_pos_mid2[0], pos_obj2[1], pos_mid2[2]]))
        self.subgoals.append(np.array([nsd_pos_mid2[0], nsd_pos_mid2[1], nsd_pos_mid2[2], nsd_pos_mid1[0], nsd_pos_mid1[1] + 0.02 * self.SCALING, nsd_pos_mid1[2]]))
        self.subgoals.append(np.array([nsd_pos_mid2[0], nsd_pos_mid2[1], nsd_pos_mid2[2], target_pos[0], target_pos[1], target_pos[2]]))

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        pose = get_link_pose(self.obj_id, -1)
        return pose[0][2] > self.goal[2] + 0.01 * self.SCALING  # reduce difficulty

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # eleven waypoints executed in sequential order
        action = np.zeros(10)
        for i, waypoint in enumerate(self._waypoints):
            if self._waypoints_done[i]:
                continue
            delta_pos1 = (waypoint[0: 3] - obs['observation'][0: 3]) / 0.01 / self.SCALING
            delta_yaw1 = (waypoint[3] - obs['observation'][4]).clip(-1, 1)
            delta_pos2 = (waypoint[5: 8] - obs['observation'][7: 10]) / 0.01 / self.SCALING
            delta_yaw2 = (waypoint[8] - obs['observation'][12]).clip(-1, 1)
            if np.abs(delta_pos1).max() > 1:
                delta_pos1 /= np.abs(delta_pos1).max()
            if np.abs(delta_pos2).max() > 1:
                delta_pos2 /= np.abs(delta_pos2).max()
            scale_factor = 0.4
            delta_pos1 *= scale_factor 
            delta_pos2 *= scale_factor
            action = np.array([delta_pos1[0], delta_pos1[1], delta_pos1[2], delta_yaw1, waypoint[4],
                               delta_pos2[0], delta_pos2[1], delta_pos2[2], delta_yaw2, waypoint[9]])
            # print(' dis: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
            #     np.linalg.norm(delta_pos1), np.abs(delta_yaw1),
            #     np.linalg.norm(delta_pos2), np.abs(delta_yaw2)))
            if np.linalg.norm(delta_pos1) * 0.01 / scale_factor < 2e-3 and np.abs(delta_yaw1) < np.deg2rad(2.) \
                    and np.linalg.norm(delta_pos2) * 0.01 / scale_factor < 2e-3 and np.abs(delta_yaw2) < np.deg2rad(2.):
                self._waypoints_done[i] = True
            break
        return action

    @property
    def waypoints(self):
        return self._waypoints

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
    print(ASSET_DIR_PATH)
    env = BiPegBoard(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()