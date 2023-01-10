import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env_pegboard import PsmEnv, goal_distance
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera,
    wrap_angle
)
from surrol.tasks.ecm_env import EcmEnv, goal_distance

from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm


class MatchBoard(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    # RPOT_POS = (0.55, -0.025, 0.6781)
    # GPOT_POS = (0.55, 0.03, 0.6781)
    # POT_ORN = (1.57,0,0)
    BOARD_POS = (0.55, 0, 0.6781)
    BOARD_ORN= (1.57,0,0)
    BOARD_SCALING = 0.03
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact
    SCALING = 5.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE=3
    haptic=True

    # TODO: grasp is sometimes not stable; check how to fix it
    def __init__(self, render_mode=None, cid = -1):
        super(MatchBoard, self).__init__(render_mode, cid)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )


    def _env_setup(self):
        super(MatchBoard, self)._env_setup()
        # np.random.seed(4)  # for experiment reproduce
        self.has_object = True
        self._waypoint_goal = True
 
        # camera
        if self._render_mode == 'human':
            # reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
            #              target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
            reset_camera(yaw=89.60, pitch=-56, dist=5.98,
                         target=(-0.13, 0.03,-0.94))
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                       scaling=self.SCALING)
        self.ecm.reset_joint(self.QPOS_ECM)
        # p.setPhysicsEngineParameter(enableFileCaching=0,numSolverIterations=10,numSubSteps=128,contactBreakingThreshold=2)


        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        # physical interaction
        self._contact_approx = False

        # metal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/dot_metal_min.jpg"))
        # newmetal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/metal2.jpg"))
        # tray pad
        board = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING,
                            useFixedBase=1)
        self.obj_ids['fixed'].append(board)  # 1
        # # p.changeVisualShape(board, -1, textureUniqueId=metal)
        # red_pot = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'pot/pot.urdf'),
        #                     np.array(self.RPOT_POS) * self.SCALING,
        #                     p.getQuaternionFromEuler(self.POT_ORN),
        #                     useFixedBase=1)
        # self.obj_ids['fixed'].append(red_pot) # 2

        # green_pot = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'pot/pot.urdf'),
        #                     np.array(self.GPOT_POS) * self.SCALING,
        #                     p.getQuaternionFromEuler(self.POT_ORN),
        #                     useFixedBase=1)
        # p.changeVisualShape(green_pot,-1,rgbaColor=(0,1,0,1),specularColor=(80,80,80))
        # self.obj_ids['fixed'].append(green_pot) # 2

        # # ch4
        # yaw = (np.random.rand() - 0.5) * np.pi
        # obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'CH4/CH4_waypoints.urdf'),
        #                     (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,  # TODO: scaling
        #                      workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
        #                      workspace_limits[2][0] + 0.01),
        #                     p.getQuaternionFromEuler((0, 0, yaw)),
        #                     useFixedBase=False,
        #                     globalScaling=self.SCALING)
        # p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        # self.obj_ids['rigid'].append(obj_id)  # 0
        # self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1

        # for i in range(1,6):
            
        #     if(i<=2):
        #         pick_item = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'CH4/CH4.urdf'),
        #                     (workspace_limits[0].mean() - i/3 * 0.25,  # TODO: scaling
        #                      workspace_limits[1].mean() -i/3*0.15,
        #                      workspace_limits[2][0] + 0.05),
        #                     p.getQuaternionFromEuler((0, 0, yaw)),
        #                     useFixedBase=False,
        #                     globalScaling=self.SCALING)
        #         p.changeVisualShape(pick_item, -1, specularColor=(80, 80, 80))
        #     else:
        #         pick_item = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'CH4/CH4.urdf'),
        #                     (workspace_limits[0].mean() - i/3 * 0.1,  # TODO: scaling
        #                      workspace_limits[1].mean() +i/3*0.15,
        #                      workspace_limits[2][0] + 0.05),
        #                     p.getQuaternionFromEuler((0, 0, yaw)),
        #                     useFixedBase=False,
        #                     globalScaling=self.SCALING)
        #         p.changeVisualShape(pick_item,-1,rgbaColor=(0,1,0,1),specularColor=(80,80,80))
        match_board = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/match_board.urdf'),
                            np.array(self.BOARD_POS) * self.SCALING,
                            p.getQuaternionFromEuler(self.BOARD_ORN),
                            globalScaling=self.BOARD_SCALING,
                            useFixedBase=1)
        self.obj_ids['fixed'].append(match_board)
        for i in range(7):
            if i==3:
                continue
            fn='match_board/'+str(i)+'.urdf'
            urdf_path = os.path.join(ASSET_DIR_PATH, fn)
            if i<3:
                if i%3==0:
                    pos_offset = [-0.15,-0.075*(i%3+1)-0.15,0.06]
                else:
                    pos_offset = [-0.15,0.075*i+0.15,0.06]
            else:
                # pos_offset = [0,-0.075*(7-i)-0.15,0.06]
                if i==4:
                    pos_offset = [0,-0.075*(i%3)-0.15,0.06]
                else:
                    pos_offset = [0,0.075*(i-4)+0.15,0.06]
            obj= p.loadURDF(urdf_path,np.array(self.BOARD_POS) * self.SCALING+pos_offset,p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
            self.obj_ids['rigid'].append(obj)
        for i in range(3):
            fn='match_board/'+chr(i+ord('a'))+'.urdf'
            urdf_path = os.path.join(ASSET_DIR_PATH, fn)
            if i%3==0:
                pos_offset = [0.15,-0.075*(i%3+1)-0.15,0.06]
            else:
                pos_offset = [0.15,0.075*i+0.15,0.06]
            obj= p.loadURDF(urdf_path,np.array(self.BOARD_POS) * self.SCALING+pos_offset,p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
        # obj4= p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/4.urdf'),[0,-0.4,0.2],p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
        # obj5= p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/5.urdf'),[0,0,0.2],p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
        # obj6= p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/6.urdf'),[0,0.4,0.2],p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
        # obja= p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/a.urdf'),[0.4,-0.4,0.2],p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
        # objb= p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/b.urdf'),[0.4,0,0.2],p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
        # objc= p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/c.urdf'),[0.4,0.4,0.2],p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
            self.obj_ids['rigid'].append(obj)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['fixed'][2], 1

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.04 * self.SCALING])
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None]  # four waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        # # for physical deployment only
        # print(" -> Needle pose: {}, {}".format(np.round(pos_obj, 4), np.round(orn_obj, 4)))
        # qs = self.psm1.get_current_joint_position()
        # joint_positions = self.psm1.inverse_kinematics(
        #     (np.array(pos_obj) + np.array([0, 0, (-0.0007 + 0.0102)]) * self.SCALING,
        #      p.getQuaternionFromEuler([-90 / 180 * np.pi, -0 / 180 * np.pi, yaw])),
        #     self.psm1.EEF_LINK_INDEX)
        # self.psm1.reset_joint(joint_positions)
        # print("qs: {}".format(joint_positions))
        # print("Cartesian: {}".format(self.psm1.get_current_position()))
        # self.psm1.reset_joint(qs)

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([self.goal[0], self.goal[1],
                                       self.goal[2] + 0.0102 * self.SCALING, yaw, -0.5])  # lift up

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self._contact_approx or self.haptic is True:
            return True  # mimic the dVRL setting
        else:
            pose = get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # four waypoints executed in sequential order
        action = np.zeros(5)
        action[4] = -0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.4
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2:
                self._waypoints[i] = None
            break

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

import math
from haptic_src.test import initTouch_right, closeTouch_right, getDeviceAction_right, startScheduler, stopScheduler
from haptic_src.test import initTouch_left, closeTouch_left, getDeviceAction_left


class MatchBoardHaptic(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    # RPOT_POS = (0.55, -0.025, 0.6781)
    # GPOT_POS = (0.55, 0.03, 0.6781)
    # POT_ORN = (1.57,0,0)
    BOARD_POS = (0.55, 0, 0.6781)
    BOARD_ORN= (1.57,0,0)
    BOARD_SCALING = 0.03
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact
    SCALING = 5.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE=3
    haptic=True

    # TODO: grasp is sometimes not stable; check how to fix it
    def __init__(self, render_mode=None, cid = -1):
        super(MatchBoardHaptic, self).__init__(render_mode, cid)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        """===initialize haptic==="""
        initTouch_right()
        startScheduler()
        """======================="""

    def _env_setup(self):
        super(MatchBoardHaptic, self)._env_setup()
        # np.random.seed(4)  # for experiment reproduce
        self.has_object = True
        self._waypoint_goal = True
 
        # camera
        if self._render_mode == 'human':
            # reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
            #              target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
            reset_camera(yaw=89.60, pitch=-56, dist=5.98,
                         target=(-0.13, 0.03,-0.94))
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                       scaling=self.SCALING)
        self.ecm.reset_joint(self.QPOS_ECM)
        # p.setPhysicsEngineParameter(enableFileCaching=0,numSolverIterations=10,numSubSteps=128,contactBreakingThreshold=2)


        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        # physical interaction
        self._contact_approx = False

        # metal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/dot_metal_min.jpg"))
        # newmetal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/metal2.jpg"))
        # tray pad
        board = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING,
                            useFixedBase=1)
        self.obj_ids['fixed'].append(board)  # 1
        # # p.changeVisualShape(board, -1, textureUniqueId=metal)
        # red_pot = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'pot/pot.urdf'),
        #                     np.array(self.RPOT_POS) * self.SCALING,
        #                     p.getQuaternionFromEuler(self.POT_ORN),
        #                     useFixedBase=1)
        # self.obj_ids['fixed'].append(red_pot) # 2

        # green_pot = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'pot/pot.urdf'),
        #                     np.array(self.GPOT_POS) * self.SCALING,
        #                     p.getQuaternionFromEuler(self.POT_ORN),
        #                     useFixedBase=1)
        # p.changeVisualShape(green_pot,-1,rgbaColor=(0,1,0,1),specularColor=(80,80,80))
        # self.obj_ids['fixed'].append(green_pot) # 2

        # # ch4
        # yaw = (np.random.rand() - 0.5) * np.pi
        # obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'CH4/CH4_waypoints.urdf'),
        #                     (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,  # TODO: scaling
        #                      workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
        #                      workspace_limits[2][0] + 0.01),
        #                     p.getQuaternionFromEuler((0, 0, yaw)),
        #                     useFixedBase=False,
        #                     globalScaling=self.SCALING)
        # p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        # self.obj_ids['rigid'].append(obj_id)  # 0
        # self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1

        # for i in range(1,6):
            
        #     if(i<=2):
        #         pick_item = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'CH4/CH4.urdf'),
        #                     (workspace_limits[0].mean() - i/3 * 0.25,  # TODO: scaling
        #                      workspace_limits[1].mean() -i/3*0.15,
        #                      workspace_limits[2][0] + 0.05),
        #                     p.getQuaternionFromEuler((0, 0, yaw)),
        #                     useFixedBase=False,
        #                     globalScaling=self.SCALING)
        #         p.changeVisualShape(pick_item, -1, specularColor=(80, 80, 80))
        #     else:
        #         pick_item = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'CH4/CH4.urdf'),
        #                     (workspace_limits[0].mean() - i/3 * 0.1,  # TODO: scaling
        #                      workspace_limits[1].mean() +i/3*0.15,
        #                      workspace_limits[2][0] + 0.05),
        #                     p.getQuaternionFromEuler((0, 0, yaw)),
        #                     useFixedBase=False,
        #                     globalScaling=self.SCALING)
        #         p.changeVisualShape(pick_item,-1,rgbaColor=(0,1,0,1),specularColor=(80,80,80))
        match_board = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/match_board.urdf'),
                            np.array(self.BOARD_POS) * self.SCALING,
                            p.getQuaternionFromEuler(self.BOARD_ORN),
                            globalScaling=self.BOARD_SCALING,
                            useFixedBase=1)
        self.obj_ids['fixed'].append(match_board)
        for i in range(7):
            if i==3:
                continue
            fn='match_board/'+str(i)+'.urdf'
            urdf_path = os.path.join(ASSET_DIR_PATH, fn)
            if i<3:
                if i%3==0:
                    pos_offset = [-0.15,-0.075*(i%3+1)-0.15,0.06]
                else:
                    pos_offset = [-0.15,0.075*i+0.15,0.06]
            else:
                # pos_offset = [0,-0.075*(7-i)-0.15,0.06]
                if i==4:
                    pos_offset = [0,-0.075*(i%3)-0.15,0.06]
                else:
                    pos_offset = [0,0.075*(i-4)+0.15,0.06]
            obj= p.loadURDF(urdf_path,np.array(self.BOARD_POS) * self.SCALING+pos_offset,p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
            self.obj_ids['rigid'].append(obj)
        for i in range(3):
            fn='match_board/'+chr(i+ord('a'))+'.urdf'
            urdf_path = os.path.join(ASSET_DIR_PATH, fn)
            if i%3==0:
                pos_offset = [0.15,-0.075*(i%3+1)-0.15,0.06]
            else:
                pos_offset = [0.15,0.075*i+0.15,0.06]
            obj= p.loadURDF(urdf_path,np.array(self.BOARD_POS) * self.SCALING+pos_offset,p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
        # obj4= p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/4.urdf'),[0,-0.4,0.2],p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
        # obj5= p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/5.urdf'),[0,0,0.2],p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
        # obj6= p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/6.urdf'),[0,0.4,0.2],p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
        # obja= p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/a.urdf'),[0.4,-0.4,0.2],p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
        # objb= p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/b.urdf'),[0.4,0,0.2],p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
        # objc= p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board/c.urdf'),[0.4,0.4,0.2],p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=self.BOARD_SCALING)
            self.obj_ids['rigid'].append(obj)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['fixed'][2], 1

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.04 * self.SCALING])
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None]  # four waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        # # for physical deployment only
        # print(" -> Needle pose: {}, {}".format(np.round(pos_obj, 4), np.round(orn_obj, 4)))
        # qs = self.psm1.get_current_joint_position()
        # joint_positions = self.psm1.inverse_kinematics(
        #     (np.array(pos_obj) + np.array([0, 0, (-0.0007 + 0.0102)]) * self.SCALING,
        #      p.getQuaternionFromEuler([-90 / 180 * np.pi, -0 / 180 * np.pi, yaw])),
        #     self.psm1.EEF_LINK_INDEX)
        # self.psm1.reset_joint(joint_positions)
        # print("qs: {}".format(joint_positions))
        # print("Cartesian: {}".format(self.psm1.get_current_position()))
        # self.psm1.reset_joint(qs)

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([self.goal[0], self.goal[1],
                                       self.goal[2] + 0.0102 * self.SCALING, yaw, -0.5])  # lift up

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self._contact_approx or self.haptic is True:
            return True  # mimic the dVRL setting
        else:
            pose = get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # six waypoints executed in sequential order
        action = np.zeros(5)
        # # haptic right
        retrived_action = np.array([0, 0, 0, 0, 0], dtype = np.float32)
        getDeviceAction_right(retrived_action)
        print(f'!!!{retrived_action}')
        # retrived_action-> x,y,z, angle, buttonState(0,1,2)
        if retrived_action[4] == 2:
            action[0] = 0
            action[1] = 0
            action[2] = 0
            action[3] = 0              
        else:
            action[0] = retrived_action[2]*0.15
            action[1] = retrived_action[0]*0.15
            action[2] = retrived_action[1]*0.15
            action[3] = -retrived_action[3]/math.pi*180*0.08
        if retrived_action[4] == 0:
            action[4] = 1
        if retrived_action[4] == 1:
            action[4] = -0.5

        return action
        # """
        # Define a human expert strategy
        # """
        # # four waypoints executed in sequential order
        # action = np.zeros(5)
        # action[4] = -0.5
        # for i, waypoint in enumerate(self._waypoints):
        #     if waypoint is None:
        #         continue
        #     delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
        #     delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
        #     if np.abs(delta_pos).max() > 1:
        #         delta_pos /= np.abs(delta_pos).max()
        #     scale_factor = 0.4
        #     delta_pos *= scale_factor
        #     action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
        #     if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2:
        #         self._waypoints[i] = None
        #     break

        # return action
    def _set_action_ecm(self, action):
        action *= 0.01 * self.SCALING
        pose_rcm = self.ecm.get_current_position()
        pose_rcm[:3, 3] += action
        pos, _ = self.ecm.pose_rcm2world(pose_rcm, 'tuple')
        joint_positions = self.ecm.inverse_kinematics((pos, None), self.ecm.EEF_LINK_INDEX)  # do not consider orn
        self.ecm.move_joint(joint_positions[:self.ecm.DoF])

    def _reset_ecm_pos(self):
        self.ecm.reset_joint(self.QPOS_ECM)

    def __del__(self):
        stopScheduler()
        closeTouch_left() 
        closeTouch_right() 

if __name__ == "__main__":
    env = MatchBoardHaptic(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
