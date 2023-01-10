import numpy as np
from panda3d.core import GraphicsPipe, GraphicsOutput

from direct.showbase.ShowBase import ShowBase
from panda3d.core import FrameBufferProperties, WindowProperties
from panda3d.core import Texture, PerspectiveLens
from typing import List
import matplotlib.pyplot as plt
from panda3d.core import ConfigVariableString
# from sensors.cameras import CameraIntrinsics


ConfigVariableString('background-color', '1.0 1.0 1.0 0.0')  # sets background to white


class SceneSimulator(ShowBase):
  """Simulates a scene with image and depth cameras."""

  IMAGE_CAMERA_RENDER_ORDER = -2
  DEPTH_CAMERA_RENDER_ORDER = -2

  def __init__(self):
    """Simulate an empty scene with no cameras.
    
    Note that cameras can be added with later calls to `add_image_camera` and
    `add_depth_camera`.
    """
    ShowBase.__init__(self)
    self._image_buffers = []
    self._image_cameras = []
    self._depth_buffers = []
    self._depth_cameras = []

  def render_frame(self):
    self.graphics_engine.render_frame()

  def add_image_camera(self, size,fov, pos, hpr, name=None):
    # set up texture and graphics buffer
    window_props = WindowProperties.size(size)
    frame_buffer_props = FrameBufferProperties()
    buffer = self.graphicsEngine.make_output(self.pipe,
      f'Image Buffer [{name}]',
      self.IMAGE_CAMERA_RENDER_ORDER,
      frame_buffer_props,
      window_props,
      GraphicsPipe.BFRefuseWindow,    # don't open a window
      self.win.getGsg(),
      self.win
    )
    texture = Texture()
    buffer.addRenderTexture(texture, GraphicsOutput.RTMCopyRam)

    # set up lens according to camera intrinsics
    lens = PerspectiveLens()
    lens.set_film_size(size)
    lens.set_fov(*np.rad2deg(fov))
    
    camera = self.makeCamera(buffer, lens=lens, camName=f'Image Camera [{name}]')
    camera.reparentTo(self.render)
    camera.setPos(*pos)
    camera.setHpr(*hpr)

    self._image_buffers.append(buffer)
    self._image_cameras.append(camera)

  def get_images(self) -> List[np.ndarray]:
    """Get the images from each image camera after the most recent rendering.

    Note that `self.render_frame()` must be called separately.
    """
    images = []
    for buffer in self._image_buffers:
      tex = buffer.getTexture()
      data = tex.getRamImage()
      image = np.frombuffer(data, np.uint8)
      image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())
      image = np.flipud(image)
      images.append(image)
    return images


if __name__ == '__main__':
  sim = SceneSimulator()
  
  n_cameras = 3
  camera_radius = 6
  camera_height = 4
#   intrinsics = CameraIntrinsics.from_size_and_fov((1920, 1080), (np.pi/6, np.pi/6))
  for i in range(n_cameras):
    camera_angle = i * (2 * np.pi / n_cameras)
    pos = (camera_radius * np.sin(camera_angle), - camera_radius * np.cos(camera_angle), camera_height)
    sim.add_image_camera((1920, 1080), (np.pi/6, np.pi/6), pos, (0, 0, 0), name=str(i))
    sim._image_cameras[i].lookAt(sim.render)

  # place box in scene
  x, y, r = 0, 0, 1
  box = sim.loader.loadModel("models/box")
  box.reparentTo(sim.render)
  box.setScale(r)
  box.setPos(x-r/2,y-r/2,0)
  sim.box = box

  sim.render_frame()
  observations = sim.get_images()

  for obs in observations:
    plt.imshow(obs)
    plt.show()