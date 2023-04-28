from pybullet_rendering import BaseRenderer, RenderingPlugin
import panda3d.core as p3d
import numpy as np
from matplotlib import pyplot as plt
from surrol.gui.scene import GymEnvScene
import time,cv2
from panda3d.core import FrameBufferProperties, WindowProperties
from panda3d.core import GraphicsPipe, GraphicsOutput
from panda3d.core import Texture
from PIL import Image

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def show_rgbd_image(image, depth_image, window_name='Image window', delay=1, depth_offset=0.0, depth_scale=1.0):
    # if depth_image.dtype != np.uint8:
    #     if depth_scale is None:
    #         depth_scale = depth_image.max() - depth_image.min()
    #     if depth_offset is None:
    #         depth_offset = depth_image.min()
    #     depth_image = np.clip((depth_image - depth_offset) / depth_scale, 0.0, 1.0)
    #     depth_image = (255.0 * depth_image).astype(np.uint8)
    # depth_image = np.tile(depth_image, (1, 1, 3))
    # if image.shape[2] == 4:  # add alpha channel
    #     alpha = np.full(depth_image.shape[:2] + (1,), 255, dtype=np.uint8)
    #     depth_image = np.concatenate([depth_image, alpha], axis=-1)
    # images = np.concatenate([image, depth_image[..., :-1]], axis=1)
    # images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)  # not needed since image is already in BGR format
    cv2.imshow(window_name, image)
    cv2.imshow('right',depth_image)
    key = cv2.waitKey(delay)
    key &= 255
    if key == 27 or key == ord('q'):
        print("Pressed ESC or q, exiting")
        exit_request = True
    else:
        exit_request = False
    return exit_request

def depth_from_zbuffer(zbuffer, znear, zfar):
    """Convert OpenGL Z-buffer values to metric depth.

    Arguments:
        zbuffer {ndarray} -- Z-buffer image
        znear {float} -- near z limit
        zfar {float} -- far z limit

    Returns:
        ndarray -- metric depth
    """
    if zbuffer.dtype != np.float32:
        zbuffer = np.divide(zbuffer, np.iinfo(zbuffer.dtype).max, dtype=np.float32)

    depth = np.zeros_like(zbuffer)
    noninf = np.not_equal(zbuffer, 1.0)
    depth[noninf] = znear * zfar / (zfar - zbuffer[noninf] * (zfar - znear))
    return depth


class Panda3DSceneRenderer(BaseRenderer):
    def __init__(self):
        BaseRenderer.__init__(self)

        self.plugin = None
        self._scene = None



        p3d.ConfigVariableBool('allow-incomplete-render').set_value(False)

        self._multisamples = 32
        self._srgb_color = True
        self._show_window = True


    def build(self, cid, scene, app):
        assert self.plugin is None
        assert isinstance(scene, GymEnvScene)

        # Bind external renderer to pybullet client
        self.plugin = RenderingPlugin(cid, self)
        
        self._scene = scene

        self._app = app
        self._engine = self._app.graphics_engine
        self._pipe = self._app.pipe
        self.dr = self._app.camNode.getDisplayRegion(0)
# DUAL ECM START
        # self.dr2 = self._app.cam2.node().getDisplayRegion(0)
# DUAL ECM END
        # self._engine = p3d.GraphicsEngine.get_global_ptr()
        # self._pipe = p3d.GraphicsPipeSelection.get_global_ptr().make_default_pipe()

        self._buffer_size = None
        self._buffer = None
        self._region = None
        self._moved_tex = None
        self._color_tex = None

        self._render_ims = []

    def destroy(self):
        assert self.plugin is not None
        
        self.plugin.unload()
        self.plugin = None
        self._scene = None

        # for i, im_np in enumerate(self._render_ims):
        #     im = Image.fromarray(im_np)
        #     im.save(f"out/img_{i}.png")


    def update_scene(self, scene_graph, materials_only):
        """Update a scene using scene_graph description
        Arguments:
            scene_graph {SceneGraph} -- scene description
            materials_only {bool} -- update only shape materials
        """

        if self._scene is None:
            return
        
        self._scene._update_graph(scene_graph, materials_only)

    # def _make_buffer(self, width, height):
    #     """Make an offscreen buffer.

    #     Arguments:
    #         width {int} -- target buffer width
    #         height {int} -- target buffer height
    #     """
    
        # fb_prop = p3d.FrameBufferProperties(p3d.FrameBufferProperties.get_default())
        # fb_prop.set_multisamples(self._multisamples)
        # fb_prop.set_srgb_color(self._srgb_color)

    #     # self._buffer = self._engine.make_output(
    #     #     self._pipe, name="offscreen", sort=0,
    #     #     fb_prop=p3d.FrameBufferProperties.get_default(),
    #     #     win_prop=p3d.WindowProperties(size=(width, height)),
    #     #     flags=p3d.GraphicsPipe.BFRefuseWindow)

    #     # self.depthBuffer = self.graphicsEngine.makeOutput(
    #     #     self.pipe, "depth buffer", -2,
    #     #     fbprops, winprops,
    #     #     GraphicsPipe.BFRefuseWindow,
    #     #     self.win.getGsg(), self.win)

    #     self._region = self._buffer.make_display_region()

    #     # self._depth_tex = p3d.Texture()
    #     # self._depth_tex.setFormat(p3d.Texture.FDepthComponent)
    #     # self._buffer.addRenderTexture(
    #     #     self._depth_tex, p3d.GraphicsOutput.RTMCopyRam, p3d.GraphicsOutput.RTPDepth)

    #     self._color_tex = p3d.Texture()
    #     self._color_tex.setFormat(p3d.Texture.FRgba8)
    #     self._buffer.addRenderTexture(
    #         self._color_tex, p3d.GraphicsOutput.RTMCopyRam, p3d.GraphicsOutput.RTPColor)
        
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

        # self._scene.app.setBackgroundColor(*scene_view.bg_color)
        # return False

        # if self._callback_fn is not None:
        #     # pass result to a callback function
        #     self._callback_fn(color, depth, mask)
        #     return False

        # send rendered results to bullet
        # if color is not None:
        #     frame.color_img[:] = color
        # if depth is not None:
        #     frame.depth_img[:] = depth
        # if mask is not None:
        #     frame.mask_img[:] = mask

        # color = self._scene.app.get_camera_image()
        # depth = self._scene.app.get_camera_depth_image()
        # if color is not None:
        #     frame.color_img[:] = color
        # if depth is not None:
        #     frame.depth_img[:] = depth

        # width, height = scene_view.viewport

        # if self._buffer is None:
        #     self._make_buffer(width, height)
        # elif self._buffer.get_size() != (width, height):
        #     self._remove_buffer()
        #     self._make_buffer(width, height)

        # if self._region.camera != self._scene.cam:
        #     self._region.camera = self._scene.cam
        #     # TODO: shadows don't appear without this dummy rendering
        #     self._engine.render_frame()


        # self._buffer.set_clear_color(scene_view.bg_color)

        # self._engine.render_frame()

        # data = self._color_tex.getRamImageAs('RGBA')
        # color_image = np.frombuffer(data, np.uint8)
        # color_image.shape = (height, width, 4)
        # color_image = np.flipud(color_image)

        requested_format = None
        tex = self.dr.getScreenshot()
        # print(f"!!!!left{tex}")
        if requested_format is None:
            data = tex.getRamImage()
        else:
            data = tex.getRamImageAs(requested_format)
        color_image = np.frombuffer(data, np.uint8)  # use data.get_data() instead of data in python 2
        color_image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())
        color_image = np.flipud(color_image)
        color_image = adjust_gamma(color_image,1.8)
# DUAL ECM STRAT
        # movedTex = self.dr2.getScreenshot()
        # # print(f"!!!!right{tex}")
        # if requested_format is None:
        #     data = movedTex.getRamImage()
        # else:
        #     data = movedTex.getRamImageAs(requested_format)        
        # moved_image = np.frombuffer(data, np.uint8)
        # moved_image.shape = (movedTex.getYSize(), movedTex.getXSize(), movedTex.getNumComponents())
        # moved_image = np.flipud(moved_image)
        # moved_image = adjust_gamma(moved_image,1.8)
# DUAL ECM END
# SHOW TWO ECM OUTPUT START
        # show_rgbd_image(color_image,moved_image)
# SHOW TWO ECM OUPUT END
        # plt.imshow(color_image)
        # plt.show()
        # time.sleep(2)

        # data = self._depth_tex.getRamImage()
        # depth_image = np.frombuffer(data, np.float32)
        # depth_image.shape = height, width
        # lens = self._scene.camera.node().get_lens()
        # depth_image = depth_from_zbuffer(depth_image, lens.near, lens.far)
        # depth_image = np.flipud(depth_image)

        # if color_image is not None:
        #     frame.color_img[:] = color_image
        # if depth_image is not None:
        #     frame.depth_img[:] = depth_image
        # if mask is not None:
        #     frame.mask_img[:] = mask
        
        return True