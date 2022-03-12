import numpy as np

from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import *
from panda3d.core import TextNode, AmbientLight, DirectionalLight, Spotlight, PerspectiveLens

from surrol.gui.scene import Scene, GymEnvScene
from surrol.gui.application import Application
from surrol.tasks.needle_pick import NeedlePick
from surrol.tasks.peg_transfer import PegTransfer


app = None
text_font = None


class StartPage(Scene):
    def __init__(self):
        super(StartPage, self).__init__()

        global text_font

        self.title = OnscreenText(text='Please select a scene:', parent=self.gui,
            pos=(0, 0.6), scale=0.1, fg=(1, 1, 1, 1),
            align=TextNode.ACenter, mayChange=1)
        self.title.setFont(text_font)

        self.btn1 = DirectButton(text="Needle Pick", parent=self.gui,
            scale=.1, command=lambda: open_scene(1), text_font=text_font,
            frameSize = (-3.7, 3.7, -2., 2.), pos=(0, 0, -0.25))

        self.btn2 = DirectButton(text="Peg Transfer", parent=self.gui,
            scale=.1, command=lambda: open_scene(2), text_font=text_font,
            frameSize = (-3.7, 3.7, -2., 2.), pos=(0, 0, 0.25))


class SurgicalTrainingCase(GymEnvScene):
    def __init__(self, env_type, env_params):
        super(SurgicalTrainingCase, self).__init__(env_type, env_params)

        global text_font

        self.back_btn = DirectButton(text="Back to Menu", parent=self.gui,
            scale=0.1, command=lambda: open_scene(0), text_font=text_font,
            frameSize = (-3.7, 3.7, -1, 1.2), pos=(-1.0, 0, 0.85))

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

    def before_simulation_step(self):
        self.env._set_action(self.psm1_action)

    def setPsmAction(self, dim, val):
        self.psm1_action[dim] = val
        
    def addPsmAction(self, dim, incre):
        self.psm1_action[dim] += incre

    def on_env_created(self):
        """Setup extrnal lights"""

        table_pos = np.array(self.env.POSE_TABLE[0]) * self.env.SCALING

        # ambient light
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.world3d.attachNewNode(alight)
        self.world3d.setLight(alnp)

        # directional light
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.4, 0.4, 0.25, 1))
        dlight.setShadowCaster(True, app.configs.shadow_resolution, app.configs.shadow_resolution)
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


def load_font():
    global app, text_font
    text_font = app.loader.loadFont('fonts/Fredoka/Fredoka-VariableFont_wdth,wght.ttf')
    text_font.setPixelsPerUnit(40)

        
def open_scene(id):
    global app

    scene = None

    if id == 0:
        scene = StartPage()
    elif id == 1:
        scene = SurgicalTrainingCase(NeedlePick, {'render_mode': 'human'})
    elif id == 2:
        scene = SurgicalTrainingCase(PegTransfer, {'render_mode': 'human'})
    
    if scene:
        app.play(scene)


app = Application()
print('Press <W><A><S><D><E><Q><Space> to control the PSM.')
load_font()
open_scene(0)
app.run()