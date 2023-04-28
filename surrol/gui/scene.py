import os
import numpy as np
import pybullet as pb
from direct.task import Task
import panda3d.core as p3d
from panda3d.core import (NodePath, Material, Quat, Vec3, Vec4, Mat4)
from pybullet_rendering import ShapeType
from pybullet_rendering.render.panda3d import Mesh
from pybullet_rendering.render.utils import primitive_mesh, decompose


class Scene:
    def __init__(self):
        self.world3d = NodePath('world3d')
        self.gui = NodePath('gui')

        from .application import Application
        self.app = Application.app
        self.cam = Application.app.cam
# DUAL ECM CAM START
        # self.cam2= Application.app.cam2
# DUAL ECM CAM END
        self.taskMgr = Application.app.taskMgr
        self.loader = Application.app.loader
        self.render = Application.app.render
        self.outview ={}
        self.flag=1
    def build_kivy_display_region(self, l, w, b, h):
        import kivy.config
        kivy.config.Config.set('graphics', 'width', round(self.app.configs.window_width * w))
        kivy.config.Config.set('graphics', 'height', round(self.app.configs.window_height * h))
        return self.app.win.make_display_region(l, w, b, h)
    
    def on_start(self):
        pass

    def on_destroy(self):
        pass

    def destroy(self):
        self.on_destroy()

        if self.world3d is not None:
            self.world3d.removeNode()
        if self.gui is not None:
            self.gui.removeNode()

        self.world3d = None
        self.gui = None

    def _initialize(self, **kwargs):
        self.on_start()


class GymEnvScene(Scene):
    def __init__(self, env_cls, env_params={}):
        super(GymEnvScene, self).__init__()

        self.env_nodes = {}

        self.env = None
        self.env_cls = env_cls
        self.env_params = env_params

        # Setup periodic simulation tasks
        self.time = 0
        self.taskMgr.add(self._step_simulation_task, "StepSimulationTask")
    
    def destroy(self):
        super(GymEnvScene, self).destroy()

        self.taskMgr.remove('StepSimulationTask')
        self.env.cid = -1

    def on_env_created(self):
        pass

    def before_simulation_step(self):
        pass

    def after_simulation_step(self):
        pass

    def _initialize(self, **kwargs):
        super(GymEnvScene, self)._initialize(**kwargs)

        assert self.env is None

        if 'cid' in kwargs:
            self.env_params['cid'] = kwargs['cid']
        self.env = self.env_cls(**self.env_params)

        self.on_env_created()
    
    def _step_simulation_task(self, task):
        """Step simulation
        """
        if task.time - self.time > 1 / 240.0:
            self.before_simulation_step()

            # Step simulation
            pb.stepSimulation()
            self.after_simulation_step()

            # Call trigger update scene (if necessary) and draw methods
            pb.getCameraImage(
                width=1, height=1,
                viewMatrix=self.env._view_matrix,
                projectionMatrix=self.env._proj_matrix)
            pb.setGravity(0,0,-10.0)
            # print(f"pb view matrix: {self.env._view_matrix}")
            self.time = task.time

        return Task.cont

    def _update_graph(self, scene_graph, materials_only):
        """Update a scene using scene_graph description
        Arguments:
            scene_graph {SceneGraph} -- scene description
            materials_only {bool} -- update only shape materials
        """

        for k, v in scene_graph.nodes.items():
            node = self.world3d.attachNewNode(f'node_{k:04d}')
            self.env_nodes[k] = node

            for shape in v.shapes:
                # load model
                if shape.type == ShapeType.Mesh:
                    model = self.loader.load_model(shape.mesh.filename)
                else:
                    mesh = Mesh.from_trimesh(primitive_mesh(shape))
                    model = node.attach_new_node(mesh)

                if shape.material is not None:
                    # set material
                    material = Material()
                    material.setAmbient(Vec4(*shape.material.diffuse_color))
                    material.setDiffuse(Vec4(*shape.material.diffuse_color))
                    material.setSpecular(Vec3(*shape.material.specular_color))
                    material.setShininess(5.0)
                    model.setMaterial(material, 1)

                    # if shape.material.diffuse_color[3] < 1.0:
                    #     model.set_transparency(p3d.TransparencyAttrib.M_alpha)
                
                    if shape.material.diffuse_texture:
                        filename = os.path.abspath(shape.material.diffuse_texture.filename)
                        texture = p3d.TexturePool.load_texture(filename)
                        model.set_texture(texture, 1)

                # set relative position
                model.reparentTo(node)
                model.setPos(*shape.pose.origin)
                model.setQuat(Quat(*shape.pose.quat))
                model.setScale(*shape.pose.scale)

    def _update_view(self, scene_view):
        """Apply scene state.

        Arguments:
            settings {SceneView} -- view settings, e.g. camera, light, viewport parameters
        """

        if scene_view.camera is not None:
            # print(f"scenn view:{scene_view.camera.pose_matrix}")
            yfov, znear, zfar, aspect = decompose(scene_view.camera.projection_matrix)
            # print(f"yfov:{yfov}, znear:{znear}, zfar{zfar}, aspect{aspect}")
            conv_mat = Mat4.convert_mat(p3d.CSZupRight, p3d.CSYupRight)
            self.cam.set_mat(conv_mat * Mat4(*scene_view.camera.pose_matrix.ravel(),))
            self.cam.node().get_lens().set_near_far(znear, zfar)
            self.cam.node().get_lens().set_fov(np.rad2deg(yfov*aspect), np.rad2deg(yfov))
            if self.flag:
                self.flag=0
                self.outview['pose']=scene_view.camera.pose_matrix
                self.outview['yfov'],self.outview['znear'],self.outview['zfar'],self.outview['aspect']=np.rad2deg(yfov),znear,zfar,np.rad2deg(yfov*aspect)
# DUAL ECM VIEW START
            static_track_ecm_view_out = np.array([[-0.8660253882408142, -0.25000008940696716, 0.4330127537250519, 0.0],
            [0.5000001192092896, -0.43301281332969666, 0.7499998211860657, 0.0],
            [4.4703490686970326e-08, 0.8660253286361694, 0.5000001192092896, 0.0],
            [0.33382686972618103, -0.49541640281677246, -2.541912794113159, 1.0]])
            trans_mat = np.array([[1, 0, -0.,  0], [0., 1, -0.,0.01 ], [0., 0., 1,  0],[0, 0, 0, 1]])
            result_mat = np.transpose(np.matmul(trans_mat,np.transpose(scene_view.camera.pose_matrix)))
            # self.cam2.set_mat(conv_mat * Mat4(*result_mat.ravel(),))
            # # self.cam2.set_mat(conv_mat * Mat4(*self.outview['pose'].ravel(),))
            # # print(f"{scene_view.camera.pose_matrix} {trans_mat} camera view matrix: {conv_mat * Mat4(*scene_view.camera.pose_matrix.ravel(),)}")
            # self.cam2.node().get_lens().set_near_far(self.outview['znear'],self.outview['zfar'])
            # self.cam2.node().get_lens().set_fov(self.outview['aspect'], self.outview['yfov'])
# DUAL ECM VIEW END
    def _update_state(self, scene_state):
        """Apply scene state.

        Arguments:
            scene_state {SceneState} -- transformations of all objects in the scene
        """

        for uid, node in self.env_nodes.items():
            node.set_mat(Mat4(*scene_state.matrix(uid).ravel()))

# (-0.8660253882408142, -0.25000008940696716, 0.4330127537250519, 0.0, 
# 0.5000001192092896, -0.43301281332969666, 0.7499998211860657, 0.0,
#  4.4703490686970326e-08, 0.8660253286361694, 0.5000001192092896, 0.0,
#  0.33382686972618103, -0.49541640281677246, -2.541912794113159, 1.0)
# (-0.8660253882408142, -0.25000008940696716, 0.4330127537250519, 0.0, 0.5000001192092896, -0.43301281332969666, 0.7499998211860657, 0.0, 4.4703490686970326e-08, 0.8660253286361694, 0.5000001192092896, 0.0, 0.33382686972618103, -0.49541640281677246, -2.541912794113159, 1.0)
