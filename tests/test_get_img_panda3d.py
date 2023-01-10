import numpy as np
import time
import cv2
from direct.showbase.ShowBase import ShowBase
from panda3d.core import FrameBufferProperties, WindowProperties
from panda3d.core import GraphicsPipe, GraphicsOutput
from panda3d.core import Texture
from panda3d.core import loadPrcFileData
from panda3d.core import SceneSetup
import numpy as np
from panda3d.core import GraphicsPipe, GraphicsOutput

from direct.showbase.ShowBase import ShowBase
from panda3d.core import FrameBufferProperties, WindowProperties
from panda3d.core import Texture, PerspectiveLens
from typing import List
import matplotlib.pyplot as plt
from panda3d.core import ConfigVariableString
# loadPrcFileData('', 'show-frame-rate-meter true')
# loadPrcFileData('', 'sync-video 0')


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
    images = np.concatenate([image, depth_image], axis=1)
    # images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)  # not needed since image is already in BGR format
    cv2.imshow(window_name, images)
    key = cv2.waitKey(delay)
    key &= 255
    if key == 27 or key == ord('q'):
        print("Pressed ESC or q, exiting")
        exit_request = True
    else:
        exit_request = False
    return exit_request


class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Load the environment model.
        self.scene = self.loader.loadModel("models/environment")
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)
        # Needed for camera image
        self.dr = self.camNode.getDisplayRegion(0)

        # Needed for camera depth image
        window_props = WindowProperties.getDefault()
        window_props = WindowProperties(window_props)
        window_props.setSize(self.win.getXSize(), self.win.getYSize())
        # window_props = WindowProperties.size(self.win.getXSize(), self.win.getYSize())
        frame_buffer_props = FrameBufferProperties.getDefault() #FrameBufferProperties()
        buffer = self.graphicsEngine.make_output(self.pipe,
                                                'Image Buffer',
                                                -2,
                                                frame_buffer_props,
                                                window_props,
                                                GraphicsPipe.BFRefuseWindow,    # don't open a window
                                                self.win.getGsg(),
                                                self.win
                                                )
        texture = Texture()
        buffer.addRenderTexture(texture, GraphicsOutput.RTMCopyRam)
        # lens = self.cam.node().getLens()
        # the near and far clipping distances can be changed if desired
        # lens.setNear(5.0)
        # lens.setFar(500.0)
        # self.depthCam = self.makeCamera(buffer)
        # self.depthCam.reparentTo(self.render)
        # self.depthCam.setHpr(60, 0, 0)
        # self.dr2 = self.depthCam.node().getDisplayRegion(0)

        # TODO: Scene is rendered twice: once for rgb and once for depth image.
        # How can both images be obtained in one rendering pass?

    def get_camera_image(self, requested_format=None):
        """
        Returns the camera's image, which is of type uint8 and has values
        between 0 and 255.
        The 'requested_format' argument should specify in which order the
        components of the image must be. For example, valid format strings are
        "RGBA" and "BGRA". By default, Panda's internal format "BGRA" is used,
        in which case no data is copied over.
        """
        tex = self.dr.getScreenshot()
        if requested_format is None:
            data = tex.getRamImage()
        else:
            data = tex.getRamImageAs(requested_format)
        image = np.frombuffer(data, np.uint8)  # use data.get_data() instead of data in python 2
        image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())
        image = np.flipud(image)
        return image

    def get_camera_depth_image(self):
        """
        Returns the camera's depth image, which is of type float32 and has
        values between 0.0 and 1.0.
        """
        depthTex = self.dr2.getScreenshot()
        data = depthTex.getRamImage()
        depth_image = np.frombuffer(data, np.uint8)
        depth_image.shape = (depthTex.getYSize(), depthTex.getXSize(), depthTex.getNumComponents())
        depth_image = np.flipud(depth_image)
        return depth_image


def main():
    app = MyApp()

    frames = 1800
    radius = 20
    step = 0.1
    start_time = time.time()

    for t in range(frames):
        angleDegrees = t * step
        angleRadians = angleDegrees * (np.pi / 180.0)
        app.cam.setPos(radius * np.sin(angleRadians), -radius * np.cos(angleRadians), 3)
        app.cam.setHpr(angleDegrees, 0, 0)
        # app.depthCam.setPos(radius * np.sin(1.57), -radius * np.cos(1.57), 3)
        # app.depthCam.setHpr(90, 0, 0)
        app.graphicsEngine.renderFrame()
        image = app.get_camera_image()
        # depth_image = app.get_camera_depth_image()
        show_rgbd_image(image, image)
        # print(f"scene camera:{SceneSetup().getCameraTransform()}")
    end_time = time.time()
    print("average FPS: {}".format(frames / (end_time - start_time)))


if __name__ == '__main__':
    main()