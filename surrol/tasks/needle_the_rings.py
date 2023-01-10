import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env import PsmsEnv, goal_distance
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera, 
    wrap_angle
)
from surrol.tasks.ecm_env import EcmEnv, goal_distance

from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm

class NeedleRings(PsmsEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact

    WORKSPACE_LIMITS1 = ((0.50, 0.60), (-0., 0.05), (0.676, 0.735))
    WORKSPACE_LIMITS2 = ((0.50, 0.60), (-0.05, 0.), (0.676, 0.735))
    SCALING = 5.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE=3
    #for haptic device demo  
    haptic=True
    def __init__(self, render_mode=None, cid = -1):
        super(NeedleRings, self).__init__(render_mode, cid)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )

    def _env_setup(self):
        super(NeedleRings, self)._env_setup()
        self.has_object = True

        # camera
        if self._render_mode == 'human':
            reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
                         target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                       scaling=self.SCALING)
        self.ecm.reset_joint(self.QPOS_ECM)
        # p.setPhysicsEngineParameter(enableFileCaching=0,numSolverIterations=10,numSubSteps=128,contactBreakingThreshold=2)


        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               workspace_limits[2][1])
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        workspace_limits = self.workspace_limits2
        pos = (workspace_limits[0][0],
               workspace_limits[1][0],
               workspace_limits[2][1])
        joint_positions = self.psm2.inverse_kinematics((pos, orn), self.psm2.EEF_LINK_INDEX)
        self.psm2.reset_joint(joint_positions)
        self.block_gripper = False

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING,useFixedBase=1)
        self.obj_ids['fixed'].append(obj_id)  # 1
        startOrientation = p.getQuaternionFromEuler([1.57,0,-1])
        ring_cyl00 = p.loadURDF(os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl.urdf"),[2.7,-0.2,3.405],startOrientation,globalScaling=0.03,useFixedBase=1)
        ring_cyl01 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl.urdf"),[2.7,0,3.405],startOrientation,globalScaling=0.03,useFixedBase	=1)
        ring_cyl02 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl.urdf"),[2.7,0.2,3.405],startOrientation,globalScaling=0.03,useFixedBase	=1)
        # ring_cyl10 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl.urdf"),[2.6,-0.1,3.405],startOrientation,globalScaling=0.02,useFixedBase	=1)
        # ring_cyl11 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl.urdf"),[2.6,0,3.405],startOrientation,globalScaling=0.02,useFixedBase	=1)
        # ring_cyl12 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl.urdf"),[2.6,0.1,3.405],startOrientation,globalScaling=0.02,useFixedBase=1)
        ring_cyl10 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl_yellow.urdf"),[2.93,-0.2,3.405],startOrientation,globalScaling=0.03,useFixedBase	=1)
        ring_cyl11 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl_yellow.urdf"),[2.93,0,3.405],startOrientation,globalScaling=0.03,useFixedBase	=1)
        ring_cyl12 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl_yellow.urdf"),[2.93,0.2,3.405],startOrientation,globalScaling=0.03,useFixedBase=1)
        self.obj_ids['fixed'].append(ring_cyl01)
        # needle
        # yaw = (np.random.rand() - 0.5) * np.pi
        yaw = 0.15*np.pi
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            (workspace_limits[0].mean() +  0.03,  # TODO: scaling
                             workspace_limits[1].mean() + 1.5* 0.15,
                             workspace_limits[2][0] + 0.01),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING*0.42)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1, self.obj_link2 = self.obj_ids['rigid'][0], 4, 5

    # def _set_action(self, action: np.ndarray):
    #     # simplified to a hand and drop by performing the first three steps
    #     obs = self._get_obs()
    #     if not self._waypoints_done[3]:  # 1: approach, 2: pick, 3: lift
    #         action = self.get_oracle_action(obs)
    #     super(BiPegTransfer, self)._set_action(action)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array(get_link_pose(self.obj_ids['fixed'][2], 0)[0])
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None, None, None]  # six waypoints
        pos_obj1, _ = get_link_pose(self.obj_id, self.obj_link2)
        pos_obj2, _ = get_link_pose(self.obj_id, self.obj_link1)
        pos_obj1, pos_obj2 = np.array(pos_obj1), np.array(pos_obj2)
        pos_dis = np.linalg.norm(pos_obj1 - pos_obj2)
        pitch1, pitch2 = np.deg2rad(-30), np.deg2rad(-30)
        jaw = 0.8

        pos_tip1 = (pos_obj1[0] + 0.002 * self.SCALING, pos_dis / 2, pos_obj1[2])
        orn_tip1 = p.getQuaternionFromEuler(np.deg2rad([90, -30, 90]))
        pose_tip1 = [pos_tip1, orn_tip1]
        pos_eef1, _ = self.psm1.pose_tip2eef(pose_tip1)
        pos_tip2 = (pos_obj1[0] - 0.002 * self.SCALING, - pos_dis / 2, pos_obj1[2])
        orn_tip2 = p.getQuaternionFromEuler(np.deg2rad([90, -150, 90]))
        pose_tip2 = [pos_tip2, orn_tip2]
        pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)
        self._waypoints[0] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, -jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, jaw])  # move to the middle

        pose_tip1[0] = (pos_obj1[0], pos_dis / 2, pos_obj1[2])
        pos_eef1, _ = self.psm1.pose_tip2eef(pose_tip1)
        pose_tip2[0] = (pos_obj1[0] + 0.002 * self.SCALING, - pos_dis / 2, pos_obj1[2])
        pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)
        self._waypoints[1] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, -jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, jaw])  # psm2 approach waypoint
        self._waypoints[2] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, -jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, -jaw])  # psm2 grasp
        self._waypoints[3] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, -jaw])  # psm1 release
        pose_tip1[0] = (pos_obj1[0] - 0.005 * self.SCALING, pos_dis / 2 + 0.01 * self.SCALING, pos_obj1[2])
        pos_eef1, _ = self.psm1.pose_tip2eef(pose_tip1)
        pose_tip2[0] = (pos_obj1[0] + 0.005 * self.SCALING, - pos_dis / 2, pos_obj1[2])
        pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)
        self._waypoints[4] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, -jaw])  # psm1 move middle
        pose_tip2[0] = (self.goal[0], self.goal[1], self.goal[2])
        pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)
        self._waypoints[5] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, -jaw])  # place

    def _meet_contact_constraint_requirement(self):
        """ add a contact constraint to the grasped needle to make it stable
        """
        return True

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # six waypoints executed in sequential order
        action = np.zeros(10)
        action[4], action[9] = 0.8, -0.8
        pitch_scaling = np.deg2rad(15)
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos1 = (waypoint[0: 3] - obs['observation'][0: 3]) / 0.01 / self.SCALING
            delta_pitch1 = ((waypoint[3] - obs['observation'][4]) / pitch_scaling).clip(-1, 1)
            delta_pos2 = (waypoint[5: 8] - obs['observation'][7: 10]) / 0.01 / self.SCALING
            delta_pitch2 = ((waypoint[8] - obs['observation'][11]) / pitch_scaling).clip(-1, 1)
            if np.abs(delta_pos1).max() > 1:
                delta_pos1 /= np.abs(delta_pos1).max()
            if np.abs(delta_pos2).max() > 1:
                delta_pos2 /= np.abs(delta_pos2).max()
            scale_factor = 0.5
            delta_pos1 *= scale_factor
            delta_pos2 *= scale_factor
            action = np.array([delta_pos1[0], delta_pos1[1], delta_pos1[2], delta_pitch1, waypoint[4],
                               delta_pos2[0], delta_pos2[1], delta_pos2[2], delta_pitch2, waypoint[9]])
            if np.linalg.norm(delta_pos1) * 0.01 / scale_factor < 1e-4 and np.abs(delta_pitch1) < 2. \
                    and np.linalg.norm(delta_pos2) * 0.01 / scale_factor < 1e-4 and np.abs(delta_pitch2) < 2.:
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

class NeedleRingsHaptic(PsmsEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact

    WORKSPACE_LIMITS1 = ((0.50, 0.60), (-0., 0.05), (0.676, 0.735))
    WORKSPACE_LIMITS2 = ((0.50, 0.60), (-0.05, 0.), (0.676, 0.735))
    SCALING = 5.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE=3
    #for haptic device demo  
    haptic=True
    def __init__(self, render_mode=None, cid = -1):
        super(NeedleRingsHaptic, self).__init__(render_mode, cid)
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
        initTouch_left()
        startScheduler()
        """======================="""

    def _env_setup(self):
        super(NeedleRingsHaptic, self)._env_setup()
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
        # p.setPhysicsEngineParameter(enableFileCaching=0,numSolverIterations=10,numSubSteps=128,contactBreakingThreshold=2)


        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               workspace_limits[2][1])
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        workspace_limits = self.workspace_limits2
        pos = (workspace_limits[0][0],
               workspace_limits[1][0],
               workspace_limits[2][1])
        joint_positions = self.psm2.inverse_kinematics((pos, orn), self.psm2.EEF_LINK_INDEX)
        self.psm2.reset_joint(joint_positions)
        self.block_gripper = False

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING,useFixedBase=1)
        self.obj_ids['fixed'].append(obj_id)  # 1
        startOrientation = p.getQuaternionFromEuler([1.57,0,-1])
        ring_cyl00 = p.loadURDF(os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl.urdf"),[2.7,-0.15,3.405],startOrientation,globalScaling=0.03,useFixedBase=1)
        ring_cyl01 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl.urdf"),[2.7,0,3.405],startOrientation,globalScaling=0.03,useFixedBase	=1)
        ring_cyl02 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl.urdf"),[2.7,0.15,3.405],startOrientation,globalScaling=0.03,useFixedBase	=1)
        # ring_cyl10 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl.urdf"),[2.6,-0.1,3.405],startOrientation,globalScaling=0.02,useFixedBase	=1)
        # ring_cyl11 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl.urdf"),[2.6,0,3.405],startOrientation,globalScaling=0.02,useFixedBase	=1)
        # ring_cyl12 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl.urdf"),[2.6,0.1,3.405],startOrientation,globalScaling=0.02,useFixedBase=1)
        ring_cyl10 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl_yellow.urdf"),[2.9,-0.15,3.405],startOrientation,globalScaling=0.03,useFixedBase	=1)
        ring_cyl11 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl_yellow.urdf"),[2.9,0,3.405],startOrientation,globalScaling=0.03,useFixedBase	=1)
        ring_cyl12 = p.loadURDF( os.path.join(ASSET_DIR_PATH, "ring_cyl/ring_cyl_yellow.urdf"),[2.9,0.15,3.405],startOrientation,globalScaling=0.03,useFixedBase=1)
        self.obj_ids['fixed'].append(ring_cyl01)
        # needle
        # yaw = (np.random.rand() - 0.5) * np.pi
        yaw = 0.15*np.pi
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            (workspace_limits[0].mean() +  0.03,  # TODO: scaling
                             workspace_limits[1].mean() + 1.5* 0.15,
                             workspace_limits[2][0] + 0.01),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING*0.45)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1, self.obj_link2 = self.obj_ids['rigid'][0], 4, 5

    # def _set_action(self, action: np.ndarray):
    #     # simplified to a hand and drop by performing the first three steps
    #     obs = self._get_obs()
    #     if not self._waypoints_done[3]:  # 1: approach, 2: pick, 3: lift
    #         action = self.get_oracle_action(obs)
    #     super(BiPegTransfer, self)._set_action(action)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array(get_link_pose(self.obj_ids['fixed'][2], 0)[0])
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None, None, None]  # six waypoints
        pos_obj1, _ = get_link_pose(self.obj_id, self.obj_link2)
        pos_obj2, _ = get_link_pose(self.obj_id, self.obj_link1)
        pos_obj1, pos_obj2 = np.array(pos_obj1), np.array(pos_obj2)
        pos_dis = np.linalg.norm(pos_obj1 - pos_obj2)
        pitch1, pitch2 = np.deg2rad(-30), np.deg2rad(-30)
        jaw = 0.8

        pos_tip1 = (pos_obj1[0] + 0.002 * self.SCALING, pos_dis / 2, pos_obj1[2])
        orn_tip1 = p.getQuaternionFromEuler(np.deg2rad([90, -30, 90]))
        pose_tip1 = [pos_tip1, orn_tip1]
        pos_eef1, _ = self.psm1.pose_tip2eef(pose_tip1)
        pos_tip2 = (pos_obj1[0] - 0.002 * self.SCALING, - pos_dis / 2, pos_obj1[2])
        orn_tip2 = p.getQuaternionFromEuler(np.deg2rad([90, -150, 90]))
        pose_tip2 = [pos_tip2, orn_tip2]
        pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)
        self._waypoints[0] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, -jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, jaw])  # move to the middle

        pose_tip1[0] = (pos_obj1[0], pos_dis / 2, pos_obj1[2])
        pos_eef1, _ = self.psm1.pose_tip2eef(pose_tip1)
        pose_tip2[0] = (pos_obj1[0] + 0.002 * self.SCALING, - pos_dis / 2, pos_obj1[2])
        pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)
        self._waypoints[1] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, -jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, jaw])  # psm2 approach waypoint
        self._waypoints[2] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, -jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, -jaw])  # psm2 grasp
        self._waypoints[3] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, -jaw])  # psm1 release
        pose_tip1[0] = (pos_obj1[0] - 0.005 * self.SCALING, pos_dis / 2 + 0.01 * self.SCALING, pos_obj1[2])
        pos_eef1, _ = self.psm1.pose_tip2eef(pose_tip1)
        pose_tip2[0] = (pos_obj1[0] + 0.005 * self.SCALING, - pos_dis / 2, pos_obj1[2])
        pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)
        self._waypoints[4] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, -jaw])  # psm1 move middle
        pose_tip2[0] = (self.goal[0], self.goal[1], self.goal[2])
        pos_eef2, _ = self.psm2.pose_tip2eef(pose_tip2)
        self._waypoints[5] = np.array([pos_eef1[0], pos_eef1[1], pos_eef1[2], pitch1, jaw,
                                       pos_eef2[0], pos_eef2[1], pos_eef2[2], pitch2, -jaw])  # place

    def _meet_contact_constraint_requirement(self):
        """ add a contact constraint to the grasped needle to make it stable
        """
        return True

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # eleven waypoints executed in sequential order
        action = np.zeros(10)
        # haptic left
        retrived_action = np.array([0, 0, 0, 0, 0], dtype = np.float32)
        getDeviceAction_left(retrived_action)
        # retrived_action-> x,y,z, angle, buttonState(0,1,2)
        if retrived_action[4] == 2:
            action[5+0] = 0
            action[5+1] = 0
            action[5+2] = 0
            action[5+3] = 0            
        else:
            action[5+0] = retrived_action[2]*0.15
            action[5+1] = retrived_action[0]*0.15
            action[5+2] = retrived_action[1]*0.15
            action[5+3] = -retrived_action[3]/math.pi*180*0.08
        if retrived_action[4] == 0:
            action[5+4] = 1
        if retrived_action[4] == 1:
            action[5+4] = -0.5

        # # haptic right
        retrived_action = np.array([0, 0, 0, 0, 0], dtype = np.float32)
        getDeviceAction_right(retrived_action)
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
    env = NeedleRingsHaptic(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
