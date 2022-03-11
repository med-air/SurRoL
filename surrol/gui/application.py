import pybullet as p
import pybullet_data
from panda3d.core import loadPrcFileData, WindowProperties, AntialiasAttrib
from direct.showbase.ShowBase import ShowBase
from direct.filter.CommonFilters import CommonFilters
from surrol.gui.scene import Scene, GymEnvScene
from surrol.gui.pybullet_renderer import Panda3DSceneRenderer
from surrol.const import ASSET_DIR_PATH


class ApplicationConfig:
    def __init__(self, **kwargs):
        # General config
        self.debug = kwargs.get('debug', False)

        # Rendering config
        self.srgb = kwargs.get('srgb', True)
        self.multisamples = kwargs.get('multisamples', 4)
        self.shadow_resolution = kwargs.get('shadow_resolution', 512)
        self.ambient_occlusion = kwargs.get('ambient_occlusion', False)

        # Window config
        self.window_width = kwargs.get('window_width', 640)
        self.window_height = kwargs.get('window_height', 480)
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

    def __init__(self, cfg=ApplicationConfig.defaults()):
        loadPrcFileData(
            "",
            f"""
            framebuffer-srgb  {1 if cfg.srgb else 0}
            framebuffer-multisample {1 if cfg.multisamples else 0}
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
        if cfg.ambient_occlusion:
            filters = CommonFilters(self.win, self.cam)
            filters.setAmbientOcclusion()

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

    def play(self, scene):
        assert isinstance(scene, Scene)

        # Recover initial scene settings
        self.setBackgroundColor(self._init_settings['bg_color'])
        self.cam.set_mat(self._init_settings['cam_pose'])
        self.cam.node().get_lens().set_near_far(self._init_settings['cam_lens'][0], self._init_settings['cam_lens'][1])
        self.cam.node().get_lens().set_fov(self._init_settings['cam_lens'][2])

        # Remove old scene
        if self.main_scene is not None:
            if isinstance(self.main_scene, GymEnvScene):
                self._renderer_if.destroy()
                p.resetSimulation()
                p.disconnect(self.pybullet_cid)
            
            self.main_scene.destroy()
        
        # Initialize new scene
        if isinstance(scene, GymEnvScene):
            self.pybullet_cid = p.connect(p.DIRECT)
            self._renderer_if.build(self.pybullet_cid, scene)

            scene._initialize(cid=self.pybullet_cid)
        
        # Attach new scene
        self.main_scene = scene
        self.main_scene.world3d.reparentTo(self.render)
        self.main_scene.gui.reparentTo(self.aspect2d)
