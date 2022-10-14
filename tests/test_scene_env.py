import argparse
import numpy as np
import math


from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import *
from panda3d.core import TextNode, AmbientLight, DirectionalLight, Spotlight, PerspectiveLens

from surrol.gui.scene import Scene, GymEnvScene
from surrol.gui.application import Application
from surrol.tasks.needle_pick import NeedlePick
from surrol.tasks.peg_transfer import PegTransfer
# from surrol.tasks.ecm_static_track import StaticTrack

from haptic_src.test import initTouch_right, closeTouch_right, getDeviceAction_right,startScheduler, stopScheduler

###########################
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
from surrol.robots.ecm import Ecm
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera
)

class StaticTrack(EcmEnv):
    # ACTION_MODE = 'cVc'
    ACTION_MODE = 'dmove'
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


import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env import PsmEnv, goal_distance
from surrol.utils.pybullet_utils import (
    get_link_pose,
    wrap_angle
)
from surrol.const import ASSET_DIR_PATH


class PegTransferWithEcm(PsmEnv):
    POSE_BOARD = ((0.55, 0, 0.6861), (0, 0, 0))  # 0.675 + 0.011 + 0.001
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.686, 0.745))
    SCALING = 5.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE=3

    # TODO: grasp is sometimes not stable; check how to fix it

    def __init__(self, render_mode=None, cid = -1):
        super(PegTransferWithEcm, self).__init__(render_mode, cid)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )

    def _env_setup(self):
        super(PegTransferWithEcm, self)._env_setup()
        self.has_object = True

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               workspace_limits[2][1])
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False

        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                       scaling=self.SCALING)
        self.ecm.reset_joint(self.QPOS_ECM)

        # peg board
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'peg_board/peg_board.urdf'),
                            np.array(self.POSE_BOARD[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_BOARD[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1
        self._pegs = np.arange(12)
        np.random.shuffle(self._pegs[:6])
        np.random.shuffle(self._pegs[6: 12])

        # blocks
        num_blocks = 4
        # for i in range(6, 6 + num_blocks):
        for i in self._pegs[6: 6 + num_blocks]:
            pos, orn = get_link_pose(self.obj_ids['fixed'][1], i)
            yaw = (np.random.rand() - 0.5) * np.deg2rad(60)
            obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'block/block.urdf'),
                                np.array(pos) + np.array([0, 0, 0.03]),
                                p.getQuaternionFromEuler((0, 0, yaw)),
                                useFixedBase=False,
                                globalScaling=self.SCALING)
            self.obj_ids['rigid'].append(obj_id)
        self._blocks = np.array(self.obj_ids['rigid'][-num_blocks:])
        np.random.shuffle(self._blocks)
        for obj_id in self._blocks[:1]:
            # change color to red
            p.changeVisualShape(obj_id, -1, rgbaColor=(255 / 255, 69 / 255, 58 / 255, 1))
        self.obj_id, self.obj_link1 = self._blocks[0], 1
    
    def _set_action_ecm(self, action):
        action *= 0.01 * self.SCALING
        pose_rcm = self.ecm.get_current_position()
        pose_rcm[:3, 3] += action
        pos, _ = self.ecm.pose_rcm2world(pose_rcm, 'tuple')
        joint_positions = self.ecm.inverse_kinematics((pos, None), self.ecm.EEF_LINK_INDEX)  # do not consider orn
        self.ecm.move_joint(joint_positions[:self.ecm.DoF])

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        # TODO: may need to tune parameters
        return np.logical_and(
            goal_distance(achieved_goal[..., :2], desired_goal[..., :2]) < 5e-3 * self.SCALING,
            np.abs(achieved_goal[..., -1] - desired_goal[..., -1]) < 4e-3 * self.SCALING
        ).astype(np.float32)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array(get_link_pose(self.obj_ids['fixed'][1], self._pegs[0])[0])
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
        pos_peg = get_link_pose(self.obj_ids['fixed'][1],
                                self._pegs[self.obj_id - np.min(self._blocks) + 6])[0]  # 6 pegs
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

###########################


app = None


class SurgicalTrainingCase(GymEnvScene):
    def __init__(self, env_type, env_params):
        super(SurgicalTrainingCase, self).__init__(env_type, env_params)

        self.psm1_action = np.zeros(env_type.ACTION_SIZE)
        self.psm1_action[4] = 0.5

        self.app.accept('w-up', self.setPsmAction, [2, 0])
        self.app.accept('w-repeat', self.addPsmAction, [2, 0.01])
        self.app.accept('s-up', self.setPsmAction, [2, 0])
        self.app.accept('s-repeat', self.addPsmAction, [2, -0.01])
        self.app.accept('d-up', self.setPsmAction, [1, 0])
        self.app.accept('d-repeat', self.addPsmAction, [1, 0.01])
        self.app.accept('a-up', self.setPsmAction, [1, 0])
        self.app.accept('a-repeat', self.addPsmAction, [1, -0.01])
        self.app.accept('q-up', self.setPsmAction, [0, 0])
        self.app.accept('q-repeat', self.addPsmAction, [0, 0.01])
        self.app.accept('e-up', self.setPsmAction, [0, 0])
        self.app.accept('e-repeat', self.addPsmAction, [0, -0.01])
        self.app.accept('space-up', self.setPsmAction, [4, 1.0])
        self.app.accept('space-repeat', self.setPsmAction, [4, -0.5])

        self.ecm_view = 0
        self.ecm_view_out = None

        self.ecm_action = np.zeros(env_type.ACTION_ECM_SIZE)
        self.app.accept('i-up', self.setEcmAction, [2, 0])
        self.app.accept('i-repeat', self.addEcmAction, [2, 0.2])
        self.app.accept('k-up', self.setEcmAction, [2, 0])
        self.app.accept('k-repeat', self.addEcmAction, [2, -0.2])
        self.app.accept('o-up', self.setEcmAction, [1, 0])
        self.app.accept('o-repeat', self.addEcmAction, [1, 0.2])
        self.app.accept('u-up', self.setEcmAction, [1, 0])
        self.app.accept('u-repeat', self.addEcmAction, [1, -0.2])
        self.app.accept('j-up', self.setEcmAction, [0, 0])
        self.app.accept('j-repeat', self.addEcmAction, [0, 0.2])
        self.app.accept('l-up', self.setEcmAction, [0, 0])
        self.app.accept('l-repeat', self.addEcmAction, [0, -0.2])
        self.app.accept('m-up', self.toggleEcmView)

    def before_simulation_step(self):

        retrived_action = np.array([0, 0, 0, 0, 0], dtype = np.float32)
        getDeviceAction_right(retrived_action)
        # print("retried action:",retrived_action)
        """retrived_action-> x,y,z, angle, buttonState(0,1,2)"""

        if retrived_action[4] == 2:
            '''Clutch'''
            self.psm1_action[0] = 0
            self.psm1_action[1] = 0
            self.psm1_action[2] = 0
            self.psm1_action[3] = 0
            
        elif retrived_action[4] != 3:
            '''Control PSM'''
            self.psm1_action[0] = retrived_action[2]
            self.psm1_action[1] = retrived_action[0]
            self.psm1_action[2] = retrived_action[1]
            self.psm1_action[3] = -retrived_action[3]/math.pi*180
            

        '''Grasping'''
        if retrived_action[4] == 1:
            self.psm1_action[4] = -0.5
        else:
            self.psm1_action[4] = 1

        '''Control ECM'''
        if retrived_action[4] == 3:
            self.ecm_action[0] = -retrived_action[0]*0.5
            self.ecm_action[1] = -retrived_action[1]*0.5
            self.ecm_action[2] = retrived_action[2]*0.5
            
        self.env._set_action(self.psm1_action)

        # self.env._set_action(self.ecm_action)
        self.env._set_action_ecm(self.ecm_action)
        self.env.ecm.render_image()

        if self.ecm_view:
            self.env._view_matrix = self.env.ecm.view_matrix
        else:
            self.env._view_matrix = self.ecm_view_out

    def setPsmAction(self, dim, val):
        self.psm1_action[dim] = val
        
    def addPsmAction(self, dim, incre):
        self.psm1_action[dim] += incre

    def addEcmAction(self, dim, incre):
        self.ecm_action[dim] += incre

    def setEcmAction(self, dim, val):
        self.ecm_action[dim] = val
    
    def toggleEcmView(self):
        self.ecm_view = not self.ecm_view

    def on_env_created(self):
        """initialize the haptic device"""
        initTouch_right()
        startScheduler()

        """Setup extrnal lights"""

        self.ecm_view_out = self.env._view_matrix

        table_pos = np.array(self.env.POSE_TABLE[0]) * self.env.SCALING
        # table_pos = np.array([0., 0., 0.])

        # ambient light
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.world3d.attachNewNode(alight)
        self.world3d.setLight(alnp)

        # directional light
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.4, 0.4, 0.25, 1))
        # dlight.setColor((0.8, 0.8, 0.8, 1))
        dlight.setShadowCaster(False, app.configs.shadow_resolution, app.configs.shadow_resolution)
        dlnp = self.world3d.attachNewNode(dlight)
        dlnp.setPos(*(table_pos + np.array([1.0, 0.0, 15.0])))
        dlnp.lookAt(*table_pos)
        self.world3d.setLight(dlnp)

        # spotlight
        slight = Spotlight('slight')
        slight.setColor((0.5, 0.5, 0.5, 1.0))
        lens = PerspectiveLens()
        lens.setNearFar(0.5, 5)
        slight.setLens(lens)
        slight.setShadowCaster(True, app.configs.shadow_resolution, app.configs.shadow_resolution)
        slnp = self.world3d.attachNewNode(slight)
        slnp.setPos(*(table_pos + np.array([0, 0.0, 5.0])))
        slnp.lookAt(*(table_pos + np.array([0.5, 0, 1.0])))
        self.world3d.setLight(slnp)

    def on_destroy(self):
        # close the haptic device
        stopScheduler()
        closeTouch_right()

        
def open_scene():
    global app

    scene = SurgicalTrainingCase(PegTransferWithEcm, {'render_mode': 'human'})
    app.play(scene)

    # Access gym env
    # scene.env
    # scene.env._set_action(...)


app = Application()
print('Press <W><A><S><D><E><Q><Space> to control the PSM.')
open_scene()
app.run()