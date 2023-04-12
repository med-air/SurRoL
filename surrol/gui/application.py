import pybullet as p
import numpy as np
import pybullet_data
import panda3d.core as p3d

from panda3d.core import loadPrcFileData, WindowProperties, AntialiasAttrib
from direct.showbase.ShowBase import ShowBase
from direct.filter.CommonFilters import CommonFilters
from surrol.gui.scene import Scene, GymEnvScene
from surrol.gui.pybullet_renderer import Panda3DSceneRenderer
from surrol.const import ASSET_DIR_PATH
import numpy as np
import time
import cv2
from direct.showbase.ShowBase import ShowBase
from panda3d.core import FrameBufferProperties, WindowProperties
from panda3d.core import GraphicsPipe, GraphicsOutput
from panda3d.core import Texture
from panda3d.core import loadPrcFileData
from panda3d.core import (NodePath, Material, Quat, Vec3, Vec4, Mat4)


class ApplicationConfig:
    def __init__(self, **kwargs):
        # General config
        self.debug = kwargs.get('debug', False)

        # Rendering config
        self.srgb = kwargs.get('srgb', True)
        self.multisamples = kwargs.get('multisamples', 2)
        self.shadow_resolution = kwargs.get('shadow_resolution', 512)
        self.ambient_occlusion = kwargs.get('ambient_occlusion', False)

        # Window config
        self.window_width = kwargs.get('window_width', 640)
        self.window_height = kwargs.get('window_height', 512)
        self.window_title = kwargs.get('window_title', 'SurRol Simulator')
    
    @staticmethod
    def defaults():
        return ApplicationConfig()
    
    @staticmethod
    def from_argparse_namespace(args):
        return ApplicationConfig(**vars(args))


class Application(ShowBase):
    # Global handler of the Application instance
    app = None
            # framebuffer-srgb true
    def __init__(self, cfg=ApplicationConfig.defaults()):
        loadPrcFileData(
            "",
            f"""
            framebuffer-stencil true
            framebuffer-multisample 1
            multisamples {cfg.multisamples}
            model-path {pybullet_data.getDataPath()}
            model-path {ASSET_DIR_PATH}
            show-frame-rate-meter 1
            gl-compile-and-execute 1
            gl-use-bindless-texture 1
            prefer-texture-buffer 1
            audio-library-name null
            """)

        super(Application, self).__init__()

        self.configs = cfg

        # Setup filters
        filters = CommonFilters(self.win, self.cam)
        if cfg.ambient_occlusion:
            filters.setAmbientOcclusion()
        if cfg.srgb:
            filters.setSrgbEncode()
            filters.setGammaAdjust(1.2)

        # Debug: "out-of-body experience" mouse-interface mode
        if cfg.debug:
            self.oobe()
        else:
            self.disableMouse()

        # Cache initial scene settings
        self._init_settings = {
            'bg_color': self.getBackgroundColor(),
            'cam_pose': self.cam.getMat(),
            'cam_lens': (self.cam.node().get_lens().get_near(),
                        self.cam.node().get_lens().get_far(),
                        self.cam.node().get_lens().get_fov())
        }

        # Window properties
        properties = WindowProperties()
        properties.setSize(cfg.window_width, cfg.window_height)
        properties.setFixedSize(True)
        properties.setTitle(cfg.window_title)
        self.win.requestProperties(properties)

        # Rendering settings
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.render.setDepthOffset(1)
        self.render.setShaderAuto()

        # The main scene instance
        self.main_scene = None 
        
        # Pybullet client id
        self.pybullet_cid = -1
        # Create interface for rendering simulation world
        self._renderer_if = Panda3DSceneRenderer()

        # Initialize a unique global handler of the Application instance
        if Application.app is None:
            Application.app = self
        else:
            raise Exception('Application instance must be unique!')
        # Needed for camera image
        # self.dr = self.camNode.getDisplayRegion(0)

        # # Needed for camera depth image
        # winprops = WindowProperties.size(self.win.getXSize(), self.win.getYSize())
        # fbprops = FrameBufferProperties()
        # fbprops.setDepthBits(1)
        # self.depthBuffer = self.graphicsEngine.makeOutput(
        #     self.pipe, "depth buffer", -2,
        #     fbprops, winprops,
        #     GraphicsPipe.BFRefuseWindow,
        #     self.win.getGsg(), self.win)
        # self.depthTex = Texture()
        # self.depthTex.setFormat(Texture.FDepthComponent)
        # self.depthBuffer.addRenderTexture(self.depthTex,
        #     GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)
        # lens = self.cam.node().getLens()
        # # the near and far clipping distances can be changed if desired
        # # lens.setNear(5.0)
        # # lens.setFar(500.0)
        # self.depthCam = self.makeCamera(self.depthBuffer,
        #     lens=lens,
        #     scene=self.render)
        # self.depthCam.reparentTo(self.cam)

        # window_props = WindowProperties.size(1200, 900)
        # frame_buffer_props = FrameBufferProperties()
        # window_props.setSize(1024, 768)
        # window_props = WindowProperties.size(self.win.getXSize(), self.win.getYSize())
        # props = FrameBufferProperties.getDefault()
        # props.setBackBuffers(0)
        # props.setRgbColor(1)
        # frame_buffer_props = FrameBufferProperties(FrameBufferProperties.getDefault()) #FrameBufferProperties()

        # fb_prop = p3d.FrameBufferProperties(p3d.FrameBufferProperties.get_default())
        # fb_prop.set_multisamples(1)
        # fb_prop.set_srgb_color(True)
        # frame_buffer_props.setBackBuffers(0)
        # frame_buffer_props.setFloatColor(1)
        # frame_buffer_props.setSrgbColor(True)
        # print(f'!!!!!{frame_buffer_props.getSrgbColor()}')   
        # buffer = self.graphicsEngine.make_output(self.pipe,
        #                                         'Image Buffer',
        #                                         -2,
        #                                         frame_buffer_props,
        #                                         window_props,
        #                                         GraphicsPipe.BFRefuseWindow,    # don't open a window
        #                                         self.win.getGsg(),
        #                                         self.win
        #                                         )
# DUAL ECM START           
        window_props = WindowProperties.getDefault()
        window_props = WindowProperties(window_props)
        frame_buffer_props = FrameBufferProperties.getDefault()

        buffer = self.graphicsEngine.make_output(self.pipe,
                                                'Image Buffer',
                                                -2,
                                                frame_buffer_props,
                                                window_props,
                                                GraphicsPipe.BFRefuseWindow,    # don't open a window
                                                self.win.getGsg(),
                                                self.win
                                                )



        if self.win is not None:
            # Close the previous window.
            oldClearDepthActive = self.win.getClearDepthActive()
            oldClearDepth = self.win.getClearDepth()
            oldClearStencilActive = self.win.getClearStencilActive()
            oldClearStencil = self.win.getClearStencil()



        self.cam2 = self.makeCamera(buffer, clearDepth=1, lens=self.cam.node().getLens(), scene=self.render, camName='cam2')
        filters = CommonFilters(buffer, self.cam2)
        filters.setGammaAdjust(1.5)
        self.cam2.reparentTo(self.render)
# DUAL ECM END
    
        # buffer.setClearDepthActive(True)
        # buffer.setClearDepth(10000000.)

        # texture = Texture()
        # buffer.addRenderTexture(texture, GraphicsOutput.RTMCopyRam)

        # depthTex = Texture()
        # depthTex.setFormat(Texture.FDepthComponent)
        # buffer.addRenderTexture(depthTex,
        #     GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)

        # lens = self.cam.node().getLens()
        # the near and far clipping distances can be changed if desired
        # lens.setNear(5.0)
        # lens.setFar(500.0)
        # self.cam2.setPos(np.sin(1.57), -np.cos(1.57), 3)
        # self.cam2.setHpr(0, 0, 0)
        # conv_mat = Mat4.convert_mat(p3d.CSZupRight, p3d.CSYupRight)
        # trans_mat = np.array([[0.999927457901944, -0.0105417096124204, -0.00582677373880446,  0], [0.0105103315794253, 0.999930239124019, -0.00538978807312579,0 ], [0.00588318483870364, 0.00532815576255780, 0.999968498950004,  -0.000457087740545159],[0, 0, 0, 1]])
        # result_mat = np.transpose(np.matmul(trans_mat,np.transpose(self.main_scene._view_matrix)))
        # self.cam2.set_mat(self.cam.getMat())
        # self.cam2.node().get_lens().set_near_far(self.cam.node().get_lens().get_near(),self.cam.node().get_lens().get_far())
        # self.cam2.node().get_lens().set_fov(self.cam.node().get_lens().get_fov())

    def play(self, scene):
        assert isinstance(scene, Scene)

        # Recover initial scene settings
        self.setBackgroundColor(self._init_settings['bg_color'])
        self.cam.set_mat(self._init_settings['cam_pose'])
        self.cam.node().get_lens().set_near_far(self._init_settings['cam_lens'][0], self._init_settings['cam_lens'][1])
        self.cam.node().get_lens().set_fov(self._init_settings['cam_lens'][2])
        # self.cam2.set_mat(self._init_settings['cam_pose'])
        # self.cam2.node().get_lens().set_near_far(self._init_settings['cam_lens'][0], self._init_settings['cam_lens'][1])
        # self.cam2.node().get_lens().set_fov(self._init_settings['cam_lens'][2])

        # Remove old scene
        if self.main_scene is not None:
            if isinstance(self.main_scene, GymEnvScene):
                self._renderer_if.destroy()
                p.resetSimulation()
                p.disconnect(self.pybullet_cid)
            
            self.main_scene.destroy()
        
        # Initialize new scene
        init_kwargs = {}
        if isinstance(scene, GymEnvScene):
            self.pybullet_cid = p.connect(p.DIRECT)
            self._renderer_if.build(self.pybullet_cid, scene,self.app)
            init_kwargs['cid'] = self.pybullet_cid

        scene._initialize(**init_kwargs)
        
        # Attach new scene
        self.main_scene = scene
        self.main_scene.world3d.reparentTo(self.render)
        self.main_scene.gui.reparentTo(self.aspect2d)
