import argparse
import numpy as np

from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import *
from panda3d.core import TextNode, AmbientLight, DirectionalLight, Spotlight, PerspectiveLens

from surrol.gui.scene import Scene, GymEnvScene
from surrol.gui.application import Application
from surrol.tasks.needle_pick import NeedlePick
from surrol.tasks.peg_transfer import PegTransfer

import math

from haptic_src.test import initTouch_right, closeTouch_right, getDeviceAction_right,startScheduler, stopScheduler


###########################
import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.ecm_env import EcmEnv, goal_distance
from surrol.robots.psm import Psm1, Psm2
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


class PsmEnv2(PsmEnv):
    """
    Single arm env using PSM1
    Refer to Gym fetch
    https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch_env.py
    ravens
    https://github.com/google-research/ravens/blob/master/ravens/environments/environment.py
    """
    ACTION_SIZE = 5  # (dx, dy, dz, dyaw/dpitch, open/close)
    ACTION_MODE = 'yaw'
    DISTANCE_THRESHOLD = 0.005
    POSE_PSM1 = ((0.05, 0.24, 0.8524), (0, 0, -(90 + 20) / 180 * np.pi))
    QPOS_PSM1 = (0, 0, 0.10, 0, 0, 0)
    POSE_TABLE = ((0.5, 0, 0.001), (0, 0, 0))
    WORKSPACE_LIMITS1 = ((0.50, 0.60), (-0.05, 0.05), (0.675, 0.745))
    SCALING = 1.

    # QPOS_ECM = (0, 0.6, 0.04, 0)
    QPOS_ECM = (0, 0.8, 0.04, 0)
    ACTION_ECM_SIZE=3

    def __init__(self, render_mode=None, cid=None):
        super(PsmEnv2, self).__init__(render_mode, cid)

        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.345 * self.SCALING),
            distance=0.97 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )

    def _env_setup(self, has_table=True):
        # camera
        if self._render_mode == 'human':
            reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
                         target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))

        # robot
        self.psm1 = Psm1(self.POSE_PSM1[0], p.getQuaternionFromEuler(self.POSE_PSM1[1]),
                         scaling=self.SCALING)
        self.psm1_eul = np.array(p.getEulerFromQuaternion(
            self.psm1.pose_rcm2world(self.psm1.get_current_position(), 'tuple')[1]))  # in the world frame
        if self.ACTION_MODE == 'yaw':
            self.psm1_eul = np.array([np.deg2rad(-90), 0., self.psm1_eul[2]])
        elif self.ACTION_MODE == 'pitch':
            self.psm1_eul = np.array([np.deg2rad(0), self.psm1_eul[1], np.deg2rad(-90)])
        else:
            raise NotImplementedError
        self.psm2 = None
        self._contact_constraint = None
        self._contact_approx = False

        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                       scaling=self.SCALING)
        self.ecm.reset_joint(self.QPOS_ECM)

        # base table
        if has_table:
            p.loadURDF(os.path.join(ASSET_DIR_PATH, 'table/table.urdf'),
                    np.array(self.POSE_TABLE[0]) * self.SCALING,
                    p.getQuaternionFromEuler(self.POSE_TABLE[1]),
                    globalScaling=self.SCALING)

        # for goal plotting
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'sphere/sphere.urdf'),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 0

        pass  # need to implement based on every task
        # self.obj_ids

        # # only for demo
        # if len(self.actions) > 0:
        #     # record the actions
        #     folder_path = os.path.join(ROOT_DIR_PATH, "../logs/actions/{}".format(self.__class__.__name__))
        #     os.makedirs(folder_path, exist_ok=True)
        #     pd.DataFrame(np.array(self.actions)).to_csv(
        #         os.path.join(folder_path, "{}.csv".format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))))
        # self.actions = []
    
    def _set_action_ecm(self, action):
        action *= 0.01 * self.SCALING
        pose_rcm = self.ecm.get_current_position()
        pose_rcm[:3, 3] += action
        pos, _ = self.ecm.pose_rcm2world(pose_rcm, 'tuple')
        joint_positions = self.ecm.inverse_kinematics((pos, None), self.ecm.EEF_LINK_INDEX)  # do not consider orn
        self.ecm.move_joint(joint_positions[:self.ecm.DoF])


class NeedlePickTissueBG(PsmEnv2):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact
    SCALING = 5.

    # TODO: grasp is sometimes not stable; check how to fix it

    def _env_setup(self):
        super(NeedlePickTissueBG, self)._env_setup(has_table=False)
        # np.random.seed(4)  # for experiment reproduce
        self.has_object = True
        self._waypoint_goal = True

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

        # obj_id0 = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
        #                     np.array(self.POSE_TRAY[0]) * self.SCALING,
        #                     p.getQuaternionFromEuler(self.POSE_TRAY[1]),
        #                     globalScaling=self.SCALING)
        # self.obj_ids['fixed'].append(obj_id0)  # 1

        # scene bg
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tissues/reserve_s1.urdf'),
                             np.array([2.0, 0.35, 3.0]), p.getQuaternionFromEuler([np.pi / 2, 0, np.pi / 2]),
                             globalScaling=self.SCALING * 1.2)
        self.obj_ids['fixed'].append(obj_id)  # 1
        surgical_tex = p.loadTexture(os.path.join(ASSET_DIR_PATH, 'tissues/meshes2/tex_reserve_s1_mesh.png'))
        # p.changeVisualShape(obj_id, -1, rgbaColor=[1,1,1,1], specularColor=[0.1, 0.1, 0.1], textureUniqueId=surgical_tex, flags=0)
        p.changeVisualShape(obj_id, -1, rgbaColor=[1,0.8,0.8,1], specularColor=[0.5, 0.5, 0.5], flags=0)

        # needle
        yaw = (np.random.rand() - 0.5) * np.pi
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            (workspace_limits[0].mean() ,  # TODO: scaling
                             workspace_limits[1].mean(),
                             workspace_limits[2][0] ),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1

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
                                       pos_obj[2] + (-0.0007 - 0.002 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 - 0.002 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([self.goal[0], self.goal[1],
                                       self.goal[2] + 0.0102 * self.SCALING, yaw, -0.5])  # lift up
        
    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self._contact_approx:
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
            scale_factor = 0.48
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-2 and np.abs(delta_yaw) < 1e-4:
                self._waypoints[i] = None
            break

        return action

    def test(self, horizon=100):
        """
        Run the test simulation without any learning algorithm for debugging purposes
        """
        steps, done = 0, False
        obs = self.reset()
        while not done and steps <= horizon:
            tic = time.time()
            action = self.get_oracle_action(obs)
            print('\n -> step: {}, action: {}'.format(steps, np.round(action, 4)))
            # print('action:', action)
            obs, reward, done, info = self.step(action)

            # img = self.render('rgb_array')
            # imageio.imwrite(f'rendered_frames2/{steps:04d}.png', img)

            if isinstance(obs, dict):
                print(" -> achieved goal: {}".format(np.round(obs['achieved_goal'], 4)))
                print(" -> desired goal: {}".format(np.round(obs['desired_goal'], 4)))
            else:
                print(" -> achieved goal: {}".format(np.round(info['achieved_goal'], 4)))
            done = info['is_success'] if isinstance(obs, dict) else done
            steps += 1
            toc = time.time()
            print(" -> step time: {:.4f}".format(toc - tic))
            time.sleep(0.05)
        print('\n -> Done: {}\n'.format(done > 0))

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
        dlight.setColor((0.2, 0.2, 0.2, 1))
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

    scene = SurgicalTrainingCase(NeedlePickTissueBG, {'render_mode': 'human'})
    app.play(scene)

    # Access gym env
    # scene.env
    # scene.env._set_action(...)


app = Application()
print('Press <W><A><S><D><E><Q><Space> to control the PSM.')
open_scene()
app.run()