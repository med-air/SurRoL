from pybullet_rendering import BaseRenderer, RenderingPlugin
import panda3d.core as p3d

from surrol.gui.scene import GymEnvScene


class Panda3DSceneRenderer(BaseRenderer):
    def __init__(self):
        BaseRenderer.__init__(self)

        self.plugin = None
        self._scene = None

    def build(self, cid, scene):
        assert self.plugin is None
        assert isinstance(scene, GymEnvScene)

        # Bind external renderer to pybullet client
        self.plugin = RenderingPlugin(cid, self)
        
        self._scene = scene

    def destroy(self):
        assert self.plugin is not None
        
        self.plugin.unload()
        self.plugin = None
        self._scene = None

    def update_scene(self, scene_graph, materials_only):
        """Update a scene using scene_graph description
        Arguments:
            scene_graph {SceneGraph} -- scene description
            materials_only {bool} -- update only shape materials
        """

        if self._scene is None:
            return
        
        self._scene._update_graph(scene_graph, materials_only)

    def render_frame(self, scene_state, scene_view, frame):
        """Render a scene at scene_state with a scene_view settings
        Arguments:
            scene_state {SceneState} --  scene state, e.g. transformations of all objects
            scene_view {SceneView} -- view settings, e.g. camera, light, viewport parameters
            frame {FrameData} -- output image buffer, ignore
        """

        if self._scene is None:
            return
        
        self._scene._update_state(scene_state)
        self._scene._update_view(scene_view)

        for k, node in self._scene.env_nodes.items():
            pose = scene_state.pose(k)
            node.setPos(*pose.origin)
            node.setQuat(p3d.Quat(*pose.quat))
            node.setScale(*pose.scale)

        self._scene.app.setBackgroundColor(*scene_view.bg_color)
        return False

        # if self._callback_fn is not None:
        #     # pass result to a callback function
        #     self._callback_fn(color, depth, mask)
        #     return False

        # # send rendered results to bullet
        # if color is not None:
        #     frame.color_img[:] = color
        # if depth is not None:
        #     frame.depth_img[:] = depth
        # if mask is not None:
        #     frame.mask_img[:] = mask
        
        # return True