import os
import time
import numpy as np
import random
import pybullet as p
import pybullet_data

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

class PegBoard(PsmEnv):
    
    POSE_BOARD = ((0.55, 0, 0.6861), (0, 0, 0))  # 0.675 + 0.011 + 0.001
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.686, 0.745))
    SCALING = 5.
    rand_id = random.randint(3,4)

    QPOS_ECM = (0, 0.8, 0.04, 0)
    ACTION_ECM_SIZE=3
    #for haptic device demo
    haptic=True

    # TODO: grasp is sometimes not stable; check how to fix it

    def __init__(self, render_mode=None, cid = -1):
        super(PegBoard, self).__init__(render_mode, cid)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )

    def _env_setup(self):
        super(PegBoard, self)._env_setup()
        self.has_object = True
        # camera
        if self._render_mode == 'human':
            # reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
            #              target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
            reset_camera(yaw=89.60, pitch=-56, dist=5.98,
                         target=(-0.13, 0.03,-0.94))
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                       scaling=self.SCALING)
        self.ecm.reset_joint(self.QPOS_ECM)

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               workspace_limits[2][1])
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False

        # # peg board
        # obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'peg_board/peg_board.urdf'),
        #                     np.array(self.POSE_BOARD[0]) * self.SCALING,
        #                     p.getQuaternionFromEuler(self.POSE_BOARD[1]),
        #                     globalScaling=self.SCALING)
        # self.obj_ids['fixed'].append(obj_id)  # 1
        # self._pegs = np.arange(12)
        # np.random.shuffle(self._pegs[:6])
        # np.random.shuffle(self._pegs[6: 12])
        startPos = np.array(self.POSE_BOARD[0])*self.SCALING
        scaling=0.15
        border_offset = [0,4.8443*scaling,0]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        wood = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/wood.jpg"))
        metal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/metal.jpg"))
        newmetal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/metal2.jpg"))

        hor_board = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/horizontal_pegboard.urdf"),startPos,startOrientation,globalScaling=scaling,useFixedBase=1)
        ver_board = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/vertical_pegboard.urdf"),startPos,startOrientation,globalScaling=scaling,useFixedBase=1)
        
        p.changeVisualShape(hor_board, -1, textureUniqueId=wood)
        p.changeVisualShape(ver_board, -1, textureUniqueId=wood)
        self.obj_ids['fixed'].append(hor_board) # 1 4
        self.obj_ids['fixed'].append(ver_board) # 2 5


        upper_offset_z = [0,0,-1.6*scaling]
        upper_offset_y = [0,0.75*scaling,0]
        for i in range(2):
            upper_pos_z= list(np.array(startPos)+np.array([x*i for x in upper_offset_z]))
            for j in range(6):
                upper_pos= list(np.array(upper_pos_z)+np.array([x*j for x in upper_offset_y]))
                upper_nut = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/upper_nut.urdf"),upper_pos,startOrientation,globalScaling=scaling,useFixedBase=1)
                upper_peg = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/upper_peg.urdf"),upper_pos,startOrientation,globalScaling=scaling,useFixedBase=1)
                p.changeVisualShape(upper_nut, -1, textureUniqueId=newmetal)
                p.changeVisualShape(upper_peg, -1, textureUniqueId=newmetal)
                ring_pos,ring_orn=get_link_pose(upper_peg, -1)
                ring_pos = list(np.array(ring_pos)+np.array([0,0,-0.0]))
                ring = p.loadURDF(os.path.join(ASSET_DIR_PATH,"ring/newring.urdf"),ring_pos,startOrientation,globalScaling=scaling)
                # ring_color=[0.5*(i+0.5*j+1),0.5*j,0.16*(i+j),1]
                # p.changeVisualShape(ring,-1,rgbaColor=ring_color)
                self.obj_ids['rigid'].append(ring) #0-11
        self._rings = np.array(self.obj_ids['rigid'][-12:])
        # np.random.shuffle(self._rings)
        for obj_id in self._rings[3:4]:
            # change color to red
            p.changeVisualShape(obj_id, -1, rgbaColor=(255 / 255, 69 / 255, 58 / 255, 1))
        self.obj_id, self.obj_link1 = self._rings[0], -1

        lower_offset = [0,2.8*scaling,0]
        for i in range(2):
            lower_pos=list(np.array(startPos)+np.array([x*i for x in lower_offset]))
            lower_nut = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/lower_nut.urdf"),lower_pos,startOrientation,globalScaling=scaling,useFixedBase=1)
            lower_peg = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/lower_peg.urdf"),lower_pos,startOrientation,globalScaling=scaling,useFixedBase=1)
            p.changeVisualShape(lower_nut, -1, textureUniqueId=newmetal)
            p.changeVisualShape(lower_peg, -1, textureUniqueId=newmetal)
            self.obj_ids['fixed'].append(lower_peg) # 43,47
            print(f"lower_peg_id:{lower_peg}")
            print(self.obj_ids['fixed'])

            border_pos = list(np.array(startPos)+np.array([x*i for x in border_offset]))
            hor_border = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/hor_border.urdf"),border_pos,startOrientation,globalScaling=scaling,useFixedBase=1)
            ver_border = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/ver_border.urdf"),border_pos,startOrientation,globalScaling=scaling,useFixedBase=1)
            p.changeVisualShape(hor_border, -1, textureUniqueId=metal)
            p.changeVisualShape(ver_border, -1, textureUniqueId=metal)       
        p.changeVisualShape(upper_nut, -1, textureUniqueId=newmetal)
        p.changeVisualShape(upper_peg, -1, textureUniqueId=newmetal)
        print(f"board{self.rand_id} {self.obj_ids['fixed'][self.rand_id]}")
        
        # blocks
        # blocks
        # num_blocks = 4
        # # for i in range(6, 6 + num_blocks):
        # for i in self._pegs[6: 6 + num_blocks]:
        #     pos, orn = get_link_pose(self.obj_ids['fixed'][1], i)
        #     yaw = (np.random.rand() - 0.5) * np.deg2rad(60)
        #     obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'block/block.urdf'),
        #                         np.array(pos) + np.array([0, 0, 0.03]),
        #                         p.getQuaternionFromEuler((0, 0, yaw)),
        #                         useFixedBase=False,
        #                         globalScaling=self.SCALING)
        #     p.changeDynamics(obj_id,
        #                -1,
        #                spinningFriction=0.001,
        #                rollingFriction=0.001,
        #                linearDamping=0.0)
        #     # p.changeDynamics(obj_id, -1, ccdSweptSphereRadius=0.00000000001)
        #     self.obj_ids['rigid'].append(obj_id)
        # self._blocks = np.array(self.obj_ids['rigid'][-num_blocks:])
        # np.random.shuffle(self._blocks)
        # for obj_id in self._blocks[:1]:
        #     # change color to red
        #     p.changeVisualShape(obj_id, -1, rgbaColor=(255 / 255, 69 / 255, 58 / 255, 1))
        # self.obj_id, self.obj_link1 = self._blocks[0], 1

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        pts = p.getContactPoints()
        print("num points=", len(pts))
        # TODO: may need to tune parameters
        return np.logical_and(
            goal_distance(achieved_goal[..., :2], desired_goal[..., :2]) < 5e-3 * self.SCALING,
            np.abs(achieved_goal[..., -1] - desired_goal[..., -1]) < 4e-3 * self.SCALING
        ).astype(np.float32)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array(self.POSE_BOARD[0])*self.SCALING
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None, None, None]  # six waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.045 * self.SCALING, yaw, 0.5])  # above object
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (0.003 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (0.003 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.045 * self.SCALING, yaw, -0.5])  # lift up

        # pos_peg = get_link_pose(self.obj_ids['fixed'][1], self.obj_id - np.min(self._blocks) + 6)[0]  # 6 pegs
        pos_peg = get_link_pose(self.obj_ids['fixed'][self.rand_id], -1)[0]  # 6 pegs
        pos_place = [self.goal[0] + pos_obj[0] - pos_peg[0],
                     self.goal[1] + pos_obj[1] - pos_peg[1], self._waypoints[0][2]]  # consider offset
        self._waypoints[4] = np.array([pos_place[0], pos_place[1], pos_place[2], yaw, -0.5])  # above goal
        self._waypoints[5] = np.array([pos_place[0], pos_place[1], pos_place[2], yaw, 0.5])  # release

    def _meet_contact_constraint_requirement(self):
        return True

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # six waypoints executed in sequential order
        action = np.zeros(5)
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.7
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 2e-3 and np.abs(delta_yaw) < np.deg2rad(2.):
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

class PegBoardHaptic(PsmEnv):
    
    POSE_BOARD = ((0.55, 0, 0.676), (0, 0, 0))  # 0.675 + 0.011 + 0.001
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.686, 0.745))
    SCALING = 5.
    rand_id = random.randint(3,4)

    QPOS_ECM = (0, 0.8, 0.04, 0)
    ACTION_ECM_SIZE=3
    #for haptic device demo
    haptic=True

    # TODO: grasp is sometimes not stable; check how to fix it

    def __init__(self, render_mode=None, cid = -1):
        super(PegBoardHaptic, self).__init__(render_mode, cid)
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
        super(PegBoardHaptic, self)._env_setup()
        self.has_object = True
        # camera
        if self._render_mode == 'human':
            # reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
            #              target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
            reset_camera(yaw=89.60, pitch=-56, dist=5.98,
                         target=(-0.13, 0.03,-0.94))
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                       scaling=self.SCALING)
        self.ecm.reset_joint(self.QPOS_ECM)

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               workspace_limits[2][1])
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False

        # # peg board
        # obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'peg_board/peg_board.urdf'),
        #                     np.array(self.POSE_BOARD[0]) * self.SCALING,
        #                     p.getQuaternionFromEuler(self.POSE_BOARD[1]),
        #                     globalScaling=self.SCALING)
        # self.obj_ids['fixed'].append(obj_id)  # 1
        # self._pegs = np.arange(12)
        # np.random.shuffle(self._pegs[:6])
        # np.random.shuffle(self._pegs[6: 12])
        startPos = np.array(self.POSE_BOARD[0])*self.SCALING
        scaling=0.15
        border_offset = [0,4.8443*scaling,0]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        wood = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/wood.jpg"))
        metal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/metal.jpg"))
        newmetal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/metal2.jpg"))

        hor_board = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/horizontal_pegboard.urdf"),startPos,startOrientation,globalScaling=scaling,useFixedBase=1)
        ver_board = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/vertical_pegboard.urdf"),startPos,startOrientation,globalScaling=scaling,useFixedBase=1)
        
        p.changeVisualShape(hor_board, -1, textureUniqueId=wood)
        p.changeVisualShape(ver_board, -1, textureUniqueId=wood)
        self.obj_ids['fixed'].append(hor_board) # 1 4
        self.obj_ids['fixed'].append(ver_board) # 2 5


        upper_offset_z = [0,0,-1.6*scaling]
        upper_offset_y = [0,0.75*scaling,0]
        for i in range(2):
            upper_pos_z= list(np.array(startPos)+np.array([x*i for x in upper_offset_z]))
            for j in range(6):
                upper_pos= list(np.array(upper_pos_z)+np.array([x*j for x in upper_offset_y]))
                upper_nut = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/upper_nut.urdf"),upper_pos,startOrientation,globalScaling=scaling,useFixedBase=1)
                upper_peg = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/upper_peg.urdf"),upper_pos,startOrientation,globalScaling=scaling,useFixedBase=1)
                p.changeVisualShape(upper_nut, -1, textureUniqueId=newmetal)
                p.changeVisualShape(upper_peg, -1, textureUniqueId=newmetal)
                ring_pos,ring_orn=get_link_pose(upper_peg, -1)
                ring_pos = list(np.array(ring_pos)+np.array([0,0,-0.0]))
                ring = p.loadURDF(os.path.join(ASSET_DIR_PATH,"ring/newring.urdf"),ring_pos,startOrientation,globalScaling=scaling)
                # ring_color=[0.5*(i+0.5*j+1),0.5*j,0.16*(i+j),1]
                # p.changeVisualShape(ring,-1,rgbaColor=ring_color)
                self.obj_ids['rigid'].append(ring) #0-11
        self._rings = np.array(self.obj_ids['rigid'][-12:])
        # np.random.shuffle(self._rings)
        for obj_id in self._rings[3:4]:
            # change color to red
            p.changeVisualShape(obj_id, -1, rgbaColor=(255 / 255, 69 / 255, 58 / 255, 1))
        self.obj_id, self.obj_link1 = self._rings[0], -1

        lower_offset = [0,2.8*scaling,0]
        for i in range(2):
            lower_pos=list(np.array(startPos)+np.array([x*i for x in lower_offset]))
            lower_nut = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/lower_nut.urdf"),lower_pos,startOrientation,globalScaling=scaling,useFixedBase=1)
            lower_peg = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/lower_peg.urdf"),lower_pos,startOrientation,globalScaling=scaling,useFixedBase=1)
            p.changeVisualShape(lower_nut, -1, textureUniqueId=newmetal)
            p.changeVisualShape(lower_peg, -1, textureUniqueId=newmetal)
            self.obj_ids['fixed'].append(lower_peg) # 43,47
            print(f"lower_peg_id:{lower_peg}")
            print(self.obj_ids['fixed'])

            border_pos = list(np.array(startPos)+np.array([x*i for x in border_offset]))
            hor_border = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/hor_border.urdf"),border_pos,startOrientation,globalScaling=scaling,useFixedBase=1)
            ver_border = p.loadURDF(os.path.join(ASSET_DIR_PATH, "pegboard/ver_border.urdf"),border_pos,startOrientation,globalScaling=scaling,useFixedBase=1)
            p.changeVisualShape(hor_border, -1, textureUniqueId=metal)
            p.changeVisualShape(ver_border, -1, textureUniqueId=metal)       
        p.changeVisualShape(upper_nut, -1, textureUniqueId=newmetal)
        p.changeVisualShape(upper_peg, -1, textureUniqueId=newmetal)
        print(f"board{self.rand_id} {self.obj_ids['fixed'][self.rand_id]}")
        
        # blocks
        # blocks
        # num_blocks = 4
        # # for i in range(6, 6 + num_blocks):
        # for i in self._pegs[6: 6 + num_blocks]:
        #     pos, orn = get_link_pose(self.obj_ids['fixed'][1], i)
        #     yaw = (np.random.rand() - 0.5) * np.deg2rad(60)
        #     obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'block/block.urdf'),
        #                         np.array(pos) + np.array([0, 0, 0.03]),
        #                         p.getQuaternionFromEuler((0, 0, yaw)),
        #                         useFixedBase=False,
        #                         globalScaling=self.SCALING)
        #     p.changeDynamics(obj_id,
        #                -1,
        #                spinningFriction=0.001,
        #                rollingFriction=0.001,
        #                linearDamping=0.0)
        #     # p.changeDynamics(obj_id, -1, ccdSweptSphereRadius=0.00000000001)
        #     self.obj_ids['rigid'].append(obj_id)
        # self._blocks = np.array(self.obj_ids['rigid'][-num_blocks:])
        # np.random.shuffle(self._blocks)
        # for obj_id in self._blocks[:1]:
        #     # change color to red
        #     p.changeVisualShape(obj_id, -1, rgbaColor=(255 / 255, 69 / 255, 58 / 255, 1))
        # self.obj_id, self.obj_link1 = self._blocks[0], 1

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        pts = p.getContactPoints()
        print("num points=", len(pts))
        # TODO: may need to tune parameters
        return np.logical_and(
            goal_distance(achieved_goal[..., :2], desired_goal[..., :2]) < 5e-3 * self.SCALING,
            np.abs(achieved_goal[..., -1] - desired_goal[..., -1]) < 4e-3 * self.SCALING
        ).astype(np.float32)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array(self.POSE_BOARD[0])*self.SCALING
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None, None, None]  # six waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.045 * self.SCALING, yaw, 0.5])  # above object
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (0.003 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (0.003 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.045 * self.SCALING, yaw, -0.5])  # lift up

        # pos_peg = get_link_pose(self.obj_ids['fixed'][1], self.obj_id - np.min(self._blocks) + 6)[0]  # 6 pegs
        pos_peg = get_link_pose(self.obj_ids['fixed'][self.rand_id], -1)[0]  # 6 pegs
        pos_place = [self.goal[0] + pos_obj[0] - pos_peg[0],
                     self.goal[1] + pos_obj[1] - pos_peg[1], self._waypoints[0][2]]  # consider offset
        self._waypoints[4] = np.array([pos_place[0], pos_place[1], pos_place[2], yaw, -0.5])  # above goal
        self._waypoints[5] = np.array([pos_place[0], pos_place[1], pos_place[2], yaw, 0.5])  # release

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        pose = get_link_pose(self.obj_id, -1)
        return pose[0][2] > self.goal[2] + 0.01 * self.SCALING

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
    env = PegBoard(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
