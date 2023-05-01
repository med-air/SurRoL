import re,os
from kivy.lang import Builder
import numpy as np
import torch
import torch.nn as nn
import pybullet as p
from panda3d_kivy.mdapp import MDApp

from direct.gui.DirectGui import *
from panda3d.core import AmbientLight, DirectionalLight, Spotlight, PerspectiveLens

import math
import imageio
import time

from surrol.gui.scene import Scene, GymEnvScene
from surrol.gui.application import Application, ApplicationConfig
from surrol.tasks.ecm_misorient import MisOrient
from surrol.tasks.ecm_reach import ECMReach
from surrol.tasks.ecm_active_track import ActiveTrack
from surrol.tasks.ecm_static_track import StaticTrack
from surrol.tasks.gauze_retrieve import GauzeRetrieve
from surrol.tasks.needle_reach import NeedleReach
from surrol.tasks.needle_pick import NeedlePick
from surrol.tasks.peg_board import PegBoard
from surrol.tasks.peg_board_bimanual import BiPegBoard
from surrol.tasks.peg_transfer import PegTransfer 
from surrol.tasks.peg_transfer_RL import PegTransferRL
from surrol.tasks.needle_regrasp_bimanual import NeedleRegrasp
from surrol.tasks.peg_transfer_bimanual import BiPegTransfer
from surrol.tasks.pick_and_place import PickAndPlace
from surrol.tasks.match_board import MatchBoard
from surrol.tasks.match_board_ii import MatchBoardII 

from surrol.tasks.needle_the_rings import NeedleRings
# from surrol.tasks.match_board_ii import BiMatchBoard
from surrol.tasks.ecm_env import EcmEnv, goal_distance,reset_camera
from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.robots.ecm import Ecm

from direct.task import Task
from surrol.utils.pybullet_utils import step

app = None
hint_printed = False
resetFlag = False
def open_scene(id):
    global app, hint_printed,resetFlag

    scene = None
    menu_dict = {0:StartPage(),1:ECMPage(),2:NeedlePage(),3:PegPage(),4:PickPlacePage()}
    task_list =[NeedlePick,PegTransfer,NeedleRegrasp,BiPegTransfer,PickAndPlace,BiPegBoard,NeedleRings,MatchBoard,ECMReach,MisOrient,StaticTrack,ActiveTrack,NeedleReach,GauzeRetrieve]
    bimanual_list=[9,10,11,12,16,17,18]
    if id < 5:
        scene = menu_dict[id]
    elif id in bimanual_list:
        
        jaws=[1.0, 1.0] if id==9 or id==10 else [1.0,1.0]
        scene = SurgicalSimulatorBimanual(task_list[(id-5)//2], {'render_mode': 'human'}, jaw_states=jaws,id=id) if id %2==1 and id!=9 else \
        SurgicalSimulatorBimanual(task_list[(id-5)//2], {'render_mode': 'human'}, jaw_states=jaws,id=id,demo=1)
    else:
        if id ==15:
            scene = SurgicalSimulator(PegBoard,{'render_mode': 'human'},id)
        elif id ==8:
            scene = SurgicalSimulator(PegTransferRL,{'render_mode': 'human'},id,demo=1)
        elif id ==6:
            scene = SurgicalSimulator(NeedlePick,{'render_mode': 'human'},id,demo=1)
        else:
            scene = SurgicalSimulator(task_list[(id-5)//2],{'render_mode': 'human'},id) if id%2==1 else\
            SurgicalSimulator(task_list[(id-5)//2],{'render_mode': 'human'},id,demo=1)
    # if id == 0:
    #     scene = StartPage()
    # elif id == 1:
    #     scene = ECMPage()    
    # elif id == 2:
    #     scene = NeedlePage()
    # elif id == 3:
    #     scene = PegPage()
    # elif id == 4:
    #     scene = PickPlacePage()
    # elif id == 5:
    #     scene = SurgicalSimulator(NeedlePick, {'render_mode': 'human'},id)
    # elif id == 6:
    #     scene = SurgicalSimulator(NeedlePick, {'render_mode': 'human'},id,demo=1)
    # elif id == 7:
    #     scene = SurgicalSimulator(PegTransfer, {'render_mode': 'human'},id)
    # elif id == 8:
    #     scene = SurgicalSimulator(PegTransfer, {'render_mode': 'human'},id,demo=1)
    # elif id == 9:
    #     scene = SurgicalSimulatorBimanual(NeedleRegrasp, {'render_mode': 'human'}, jaw_states=[1.0, -0.5],id=id)
    # elif id == 10:
    #     scene = SurgicalSimulatorBimanual(NeedleRegrasp, {'render_mode': 'human'}, jaw_states=[1.0, -0.5],id=id,demo=1)
    # elif id == 11:
    #     scene = SurgicalSimulatorBimanual(BiPegTransfer, {'render_mode': 'human'}, jaw_states=[1.0, 1.0],id=id)
    # elif id == 12:
    #     scene = SurgicalSimulatorBimanual(BiPegTransfer, {'render_mode': 'human'}, jaw_states=[1.0, 1.0],id=id,demo=1)
    # elif id == 13:
    #     scene = SurgicalSimulator(PickAndPlace, {'render_mode': 'human'},id)
    # elif id == 14:
    #     scene = SurgicalSimulator(PickAndPlace, {'render_mode': 'human'},id,demo=1)
    # elif id == 15:
    #     scene = SurgicalSimulator(PegBoard, {'render_mode': 'human'},id)
    # elif id == 16:
    #     scene = SurgicalSimulator(PegBoard, {'render_mode': 'human'},id)
    # elif id == 17:
    #     scene = SurgicalSimulatorBimanual(NeedleRings, {'render_mode': 'human'}, jaw_states=[1.0, 1.0],id=id)
    # elif id == 18:
    #     scene = SurgicalSimulatorBimanual(NeedleRings, {'render_mode': 'human'}, jaw_states=[1.0, 1.0],id=id,demo=1)
    # elif id == 19:
    #     scene = SurgicalSimulator(MatchBoard, {'render_mode': 'human'},id)
    # elif id == 20:
    #     scene = SurgicalSimulator(MatchBoard, {'render_mode': 'human'},id,demo=1)
    # elif id == 21:
    #     scene = SurgicalSimulator(ECMReach, {'render_mode': 'human'},id)
    # elif id == 22:
    #     scene = SurgicalSimulator(ECMReach, {'render_mode': 'human'},id,demo=1)
    # elif id == 23:
    #     scene = SurgicalSimulator(MisOrient, {'render_mode': 'human'},id)
    # elif id == 24:
    #     scene = SurgicalSimulator(MisOrient, {'render_mode': 'human'},id,demo=1)
    # elif id == 25:
    #     scene = SurgicalSimulator(StaticTrack, {'render_mode': 'human'},id)
    # elif id == 26:
    #     scene = SurgicalSimulator(StaticTrack, {'render_mode': 'human'},id,demo=1)
    # elif id == 27:
    #     scene = SurgicalSimulator(ActiveTrack, {'render_mode': 'human'},id)
    # elif id == 28:
    #     scene = SurgicalSimulator(ActiveTrack, {'render_mode': 'human'},id,demo=1)
    # elif id == 29:
    #     scene = SurgicalSimulator(NeedleReach, {'render_mode': 'human'},id)
    # elif id == 30:
    #     scene = SurgicalSimulator(NeedleReach, {'render_mode': 'human'},id,demo=1)
    # elif id == 31:
    #     scene = SurgicalSimulator(GauzeRetrieve, {'render_mode': 'human'},id)
    # elif id == 32:
    #     scene = SurgicalSimulator(GauzeRetrieve, {'render_mode': 'human'},id,demo=1)

    # if id in (1, 2) and not hint_printed:
    #     print('Press <W><A><S><D><E><Q><Space> to control the PSM.')
    #     hint_printed = True
    
    if scene:
        app.play(scene)


selection_panel_kv = '''MDBoxLayout:
    orientation: 'vertical'
    spacing: dp(10)
    padding: dp(20)

    MDLabel:
        text: "SurRol Simulator v2"
        theme_text_color: "Primary"
        font_style: "H6"
        bold: True
        size_hint: 1.0, 0.1

    MDSeparator:

    MDGridLayout:
        cols: 2
        spacing: "30dp"
        padding: "20dp", "10dp", "20dp", "20dp"
        size_hint: 1.0, 0.9
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/ecm_track.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "1. Endoscope Fov Control"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Practise ECM control skills"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn1
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"

        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/needlepick_poster.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "2. Fundamental Actions"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Practise fundamental actions in surgery" 
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn2
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        

        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/pegtransfer_poster.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "3. Basic Robot Skill Training Tasks"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Pratise positioning and orienting objects"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0

                MDRaisedButton:
                    id: btn3
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        

'''      

        # MDCard:
        #     orientation: "vertical"
        #     size_hint: .45, None
        #     height: box_top.height + box_bottom.height

        #     MDBoxLayout:
        #         id: box_top
        #         spacing: "20dp"
        #         adaptive_height: True

        #         FitImage:
        #             source: "images/pick&place.png"
        #             size_hint: 0.5, None
        #             height: text_box.height

        #         MDBoxLayout:
        #             id: text_box
        #             orientation: "vertical"
        #             adaptive_height: True
        #             spacing: "10dp"
        #             padding: 0, "10dp", "10dp", "10dp"

        #             MDLabel:
        #                 text: "4. Pick & Place Tasks"
        #                 theme_text_color: "Primary"
        #                 font_style: "H6"
        #                 bold: True
        #                 adaptive_height: True

        #             MDLabel:
        #                 text: "Practice positioning and orienting objects"
        #                 adaptive_height: True
        #                 theme_text_color: "Primary"

        #     MDSeparator:

        #     MDBoxLayout:
        #         id: box_bottom
        #         adaptive_height: True
        #         padding: "0dp", 0, 0, 0
                
        #         MDRaisedButton:
        #             id: btn4
        #             text: "Play"
        #             size_hint: 0.8, 1.0
        #         MDIconButton:
        #             icon: "application-settings"

Peg_panel_kv = '''MDBoxLayout:
    orientation: 'vertical'


    MDLabel:
        text: "                                             Basic Robot Skill Training Tasks"
        theme_text_color: "Primary"
        font_style: "H4"
        bold: True
        spacing: "10dp"
        size_hint: 1.0, 0.3


    MDSeparator:

    MDGridLayout:
        cols: 2
        spacing: "40dp"
        padding: "20dp", "10dp", "20dp", "20dp"
        size_hint: 1.0, 1.0
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/pegtransfer_poster.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Peg Transfer"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Move the gripper to a randomly sampled position"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn1
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/bipegtransfer_poster.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Bi-Peg Transfer"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Bimanual peg transfer"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn2
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"


        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/pegboard.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Peg Board"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Transfer red ring from peg board to peg on floor"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn3
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/pick&place.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Pick and Place"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Place colored jacks into matching containers"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn4
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/matchboard.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Match Board"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Transfer various objects into matching spaces"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn5
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/needlerings.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Needle the Rings"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Pass a needle through the target ring"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0

                MDRaisedButton:
                    id: btn6
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"



    MDBoxLayout:
        MDRectangleFlatIconButton:
            icon: "exit-to-app"
            id: btn7
            text: "Exit"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1, 0
        MDRectangleFlatIconButton:
            icon: "head-lightbulb-outline"
            id: btn8
            text: "AI Assistant"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1, 0 
        MDRectangleFlatIconButton:
            icon: "chart-histogram"
            id: btn8
            text: "Evaluation"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1,0
        MDRectangleFlatIconButton:
            icon: "help-box"
            id: btn8
            text: "Help"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1,0

'''

PickPlace_panel_kv = '''MDBoxLayout:
    orientation: 'vertical'


    MDLabel:
        text: "                                                              Pick & Place Tasks"
        theme_text_color: "Primary"
        font_style: "H4"
        bold: True
        spacing: "10dp"
        size_hint: 1.0, 0.3


    MDSeparator:

    MDGridLayout:
        cols: 2
        spacing: "40dp"
        padding: "20dp", "10dp", "20dp", "20dp"
        size_hint: 1.0, 1.0
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/gauze_retrieve.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Gauze Retrieve"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: " Pick the gauze and place it at the target position"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn1
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"

        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/pick&place.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Pick and Place"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Place colored jacks into matching colored containers"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn2
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/matchboard.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Match Board"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Pick up various objects and place them into corresponding spaces on the board"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn3
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        


    MDBoxLayout:
        MDRectangleFlatIconButton:
            icon: "exit-to-app"
            id: btn4
            text: "Exit"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1, 0
        MDRectangleFlatIconButton:
            icon: "head-lightbulb-outline"
            id: btn5
            text: "AI Assistant"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1, 0 
        MDRectangleFlatIconButton:
            icon: "chart-histogram"
            id: btn6
            text: "Evaluation"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1,0
        MDRectangleFlatIconButton:
            icon: "help-box"
            id: btn7
            text: "Help"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1,0

'''

needle_panel_kv = '''MDBoxLayout:
    orientation: 'vertical'


    MDLabel:
        text: "                                                        Fundamental Actions"
        theme_text_color: "Primary"
        font_style: "H4"
        bold: True
        spacing: "10dp"
        size_hint: 1.0, 0.3


    MDSeparator:

    MDGridLayout:
        cols: 2
        spacing: "40dp"
        padding: "20dp", "10dp", "20dp", "20dp"
        size_hint: 1.0, 1.0
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/needle_reach.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Needle Reach"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Move the gripper to a randomly sampled position"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn1
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"

        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/gauze_retrieve.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Gauze Retrieve"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Pick the gauze and place it at the target position"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn2
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"

        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/needlepick_poster.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Needle Pick"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Pick up the needle and move to target position"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn3
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"


        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/needleregrasp_poster.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Needle Regrasp"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Bimanual version of needle pick"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn4
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        




    MDBoxLayout:
        MDRectangleFlatIconButton:
            icon: "exit-to-app"
            id: btn5
            text: "Exit"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1, 0
        MDRectangleFlatIconButton:
            icon: "head-lightbulb-outline"
            id: btn6
            text: "AI Assistant"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1, 0 
        MDRectangleFlatIconButton:
            icon: "chart-histogram"
            id: btn6
            text: "Evaluation"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1,0
        MDRectangleFlatIconButton:
            icon: "help-box"
            id: btn6
            text: "Help"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1,0

'''



ECM_panel_kv = '''MDBoxLayout:
    orientation: 'vertical'


    MDLabel:
        text: "                                                     Endoscope Fov Control"
        theme_text_color: "Primary"
        font_style: "H4"
        bold: True
        spacing: "10dp"
        size_hint: 1.0, 0.3


    MDSeparator:

    MDGridLayout:
        cols: 2
        spacing: "40dp"
        padding: "20dp", "10dp", "20dp", "20dp"
        size_hint: 1.0, 1.0
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/ecm_reach.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "ECM Reach"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Move the ECM to a randomly sampled position"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn1
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        
        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/misorient.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "MisOrient"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Adjust ECM to minimize misorientation"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn2
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"


        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/static_track.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Static Track"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Make ECM track a static target cube"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0
                
                MDRaisedButton:
                    id: btn3
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"
        

        MDCard:
            orientation: "vertical"
            size_hint: .45, None
            height: box_top.height + box_bottom.height

            MDBoxLayout:
                id: box_top
                spacing: "20dp"
                adaptive_height: True

                FitImage:
                    source: "images/active_track.png"
                    size_hint: 0.5, None
                    height: text_box.height

                MDBoxLayout:
                    id: text_box
                    orientation: "vertical"
                    adaptive_height: True
                    spacing: "10dp"
                    padding: 0, "10dp", "10dp", "10dp"

                    MDLabel:
                        text: "Active Track"
                        theme_text_color: "Primary"
                        font_style: "H6"
                        bold: True
                        adaptive_height: True

                    MDLabel:
                        text: "Make ECM track a active target cube"
                        adaptive_height: True
                        theme_text_color: "Primary"

            MDSeparator:

            MDBoxLayout:
                id: box_bottom
                adaptive_height: True
                padding: "0dp", 0, 0, 0

                MDRaisedButton:
                    id: btn4
                    text: "Play"
                    size_hint: 0.8, 1.0
                MDIconButton:
                    icon: "application-settings"




    MDBoxLayout:
        MDRectangleFlatIconButton:
            icon: "exit-to-app"
            id: btn5
            text: "Exit"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1, 0
        MDRectangleFlatIconButton:
            icon: "head-lightbulb-outline"
            id: btn6
            text: "AI Assistant"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1, 0 
        MDRectangleFlatIconButton:
            icon: "chart-histogram"
            id: btn6
            text: "Evaluation"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.primary_color
            size_hint: 0.1,0
        MDRectangleFlatIconButton:
            icon: "help-box"
            id: btn6
            text: "Help"
            text_color: (1, 1, 1, 1)
            icon_color: (1, 1, 1, 1)
            md_bg_color: app.theme_cls.bg_light
            size_hint: 0.1,0


'''
class SelectionUI(MDApp):

    def __init__(self, panda_app, display_region):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.screen = Builder.load_string(selection_panel_kv)
        return self.screen

    def on_start(self):
        self.screen.ids.btn1.bind(on_press = lambda _: open_scene(1))
        self.screen.ids.btn2.bind(on_press = lambda _: open_scene(2))
        self.screen.ids.btn3.bind(on_press = lambda _: open_scene(3))
        # self.screen.ids.btn4.bind(on_press = lambda _: open_scene(4))

        # self.screen.ids.btn11.bind(on_press = lambda _: open_scene(11))

class NeedleUI(MDApp):

    def __init__(self, panda_app, display_region):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.screen = Builder.load_string(needle_panel_kv)
        return self.screen

    def on_start(self):
        self.screen.ids.btn1.bind(on_press = lambda _: open_scene(29))
        self.screen.ids.btn2.bind(on_press = lambda _: open_scene(31))
        self.screen.ids.btn3.bind(on_press = lambda _: open_scene(6))
        self.screen.ids.btn4.bind(on_press = lambda _: open_scene(9))
        self.screen.ids.btn5.bind(on_press = lambda _: open_scene(0))
        self.screen.ids.btn6.bind(on_press = lambda _: open_scene(2))


class PegUI(MDApp):

    def __init__(self, panda_app, display_region):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.screen = Builder.load_string(Peg_panel_kv)
        return self.screen

    def on_start(self):
        self.screen.ids.btn1.bind(on_press = lambda _: open_scene(7))
        self.screen.ids.btn2.bind(on_press = lambda _: open_scene(11))
        self.screen.ids.btn3.bind(on_press = lambda _: open_scene(15))
        self.screen.ids.btn4.bind(on_press = lambda _: open_scene(13))
        self.screen.ids.btn5.bind(on_press = lambda _: open_scene(19))
        self.screen.ids.btn6.bind(on_press = lambda _: open_scene(17))
        self.screen.ids.btn7.bind(on_press = lambda _: open_scene(0))
        self.screen.ids.btn8.bind(on_press = lambda _: open_scene(3))

class PickPlaceUI(MDApp):

    def __init__(self, panda_app, display_region):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.screen = Builder.load_string(PickPlace_panel_kv)
        return self.screen

    def on_start(self):
        self.screen.ids.btn1.bind(on_press = lambda _: open_scene(31))
        self.screen.ids.btn2.bind(on_press = lambda _: open_scene(13))
        self.screen.ids.btn3.bind(on_press = lambda _: open_scene(19))
        self.screen.ids.btn4.bind(on_press = lambda _: open_scene(0))
        self.screen.ids.btn5.bind(on_press = lambda _: open_scene(4))
        self.screen.ids.btn6.bind(on_press = lambda _: open_scene(4))
        self.screen.ids.btn7.bind(on_press = lambda _: open_scene(4))


class ECMUI(MDApp):

    def __init__(self, panda_app, display_region):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.screen = Builder.load_string(ECM_panel_kv)
        return self.screen

    def on_start(self):
        self.screen.ids.btn1.bind(on_press = lambda _: open_scene(21))
        self.screen.ids.btn2.bind(on_press = lambda _: open_scene(23))
        self.screen.ids.btn3.bind(on_press = lambda _: open_scene(25))
        self.screen.ids.btn4.bind(on_press = lambda _: open_scene(27))
        self.screen.ids.btn5.bind(on_press = lambda _: open_scene(0))
        self.screen.ids.btn6.bind(on_press = lambda _: open_scene(1))


class StartPage(Scene):
    def __init__(self):
        super(StartPage, self).__init__()
        
    def on_start(self):
        self.ui_display_region = self.build_kivy_display_region(0, 1.0, 0, 1.0)
        self.kivy_ui = SelectionUI(
            self.app,
            self.ui_display_region
        )
        self.kivy_ui.run()
    
    def on_destroy(self):
        # !!! important
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)

class NeedlePage(Scene):
    def __init__(self):
        super(NeedlePage, self).__init__()
        
    def on_start(self):
        self.ui_display_region = self.build_kivy_display_region(0, 1.0, 0, 1.0)
        self.kivy_ui = NeedleUI(
            self.app,
            self.ui_display_region
        )
        self.kivy_ui.run()
    
    def on_destroy(self):
        # !!! important
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)

class PegPage(Scene):
    def __init__(self):
        super(PegPage, self).__init__()
        
    def on_start(self):
        self.ui_display_region = self.build_kivy_display_region(0, 1.0, 0, 1.0)
        self.kivy_ui = PegUI(
            self.app,
            self.ui_display_region
        )
        self.kivy_ui.run()
    
    def on_destroy(self):
        # !!! important
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)

class PickPlacePage(Scene):
    def __init__(self):
        super(PickPlacePage, self).__init__()
        
    def on_start(self):
        self.ui_display_region = self.build_kivy_display_region(0, 1.0, 0, 1.0)
        self.kivy_ui = PickPlaceUI(
            self.app,
            self.ui_display_region
        )
        self.kivy_ui.run()
    
    def on_destroy(self):
        # !!! important
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)

class ECMPage(Scene):
    def __init__(self):
        super(ECMPage, self).__init__()
        
    def on_start(self):
        self.ui_display_region = self.build_kivy_display_region(0, 1.0, 0, 1.0)
        self.kivy_ui = ECMUI(
            self.app,
            self.ui_display_region
        )
        self.kivy_ui.run()
    
    def on_destroy(self):
        # !!! important
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)

menu_bar_kv_haptic = '''MDBoxLayout:
    md_bg_color: (1, 0, 0, 0)
    # adaptive_height: True
    padding: "0dp", 0, 0, 0
    
    MDRectangleFlatIconButton:
        icon: "exit-to-app"
        id: btn1
        text: "Exit"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.primary_color
        size_hint: 0.25, 1.0
    MDRectangleFlatIconButton:
        icon: "head-lightbulb-outline"
        id: btn2
        text: "AI Assistant"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.bg_light
        size_hint: 0.25, 1.0
    MDRectangleFlatIconButton:
        icon: "chart-histogram"
        id: btn3
        text: "Evaluation"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.primary_color
        size_hint: 0.25, 1.0
    MDRectangleFlatIconButton:
        icon: "help-box"
        id: btn4
        text: "Switch to Demo"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.bg_light
        size_hint: 0.25, 1.0
'''
menu_bar_kv_RL = '''MDBoxLayout:
    md_bg_color: (1, 0, 0, 0)
    # adaptive_height: True
    padding: "0dp", 0, 0, 0
    
    MDRectangleFlatIconButton:
        icon: "exit-to-app"
        id: btn1
        text: "Exit"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.primary_color
        size_hint: 0.25, 1.0
    MDRectangleFlatIconButton:
        icon: "head-lightbulb-outline"
        id: btn2
        text: "AI Assistant"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.bg_light
        size_hint: 0.25, 1.0
    MDRectangleFlatIconButton:
        icon: "chart-histogram"
        id: btn3
        text: "Evaluation"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.primary_color
        size_hint: 0.25, 1.0
    MDRectangleFlatIconButton:
        icon: "help-box"
        id: btn4
        text: "Switch to Haptic Device Training"
        text_color: (1, 1, 1, 1)
        icon_color: (1, 1, 1, 1)
        md_bg_color: app.theme_cls.bg_light
        size_hint: 0.25, 1.0
'''
class MenuBarUI(MDApp):
    def __init__(self, panda_app, display_region,id = None):
        super().__init__(panda_app=panda_app, display_region=display_region)
        self.screen = None
        self.id = id
        self.ecm_list=[i for i in range(21,29)]
        self.fund_list = [31,32,5,6,9,10,29,30]
        self.basic_list=[7,8,11,12,15,16,17,18,13,14,19,20]
    def build(self):
        self.theme_cls.theme_style = "Dark"
        if self.id % 2 == 0:
            self.screen = Builder.load_string(menu_bar_kv_RL)
        else:
            self.screen = Builder.load_string(menu_bar_kv_haptic)
        return self.screen

    def on_start(self):
        # scene_menu = 1
        if self.id in self.ecm_list:
            self.screen.ids.btn1.bind(on_press = lambda _: open_scene(1))
        elif self.id in self.fund_list:
            self.screen.ids.btn1.bind(on_press = lambda _: open_scene(2))
        elif self.id in self.basic_list:
            self.screen.ids.btn1.bind(on_press = lambda _: open_scene(3))
        else:
            self.screen.ids.btn1.bind(on_press = lambda _: open_scene(0))
        if self.id % 2 ==0:
            self.screen.ids.btn4.bind(on_press = lambda _: (open_scene(0), open_scene(self.id-1)))
        else:
            self.screen.ids.btn4.bind(on_press = lambda _:(open_scene(0), open_scene(self.id+1)))
class SurgicalSimulatorBase(GymEnvScene):
    def __init__(self, env_type, env_params):
        super(SurgicalSimulatorBase, self).__init__(env_type, env_params)
    def before_simulation_step(self):
        pass

    def on_env_created(self):
        """Setup extrnal lights"""
        self.ecm_view_out = self.env._view_matrix

        table_pos = np.array(self.env.POSE_TABLE[0]) * self.env.SCALING

        # ambient light
        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.world3d.attachNewNode(alight)
        self.world3d.setLight(alnp)

        # directional light
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.4, 0.4, 0.25, 1))
        # dlight.setShadowCaster(True, app.configs.shadow_resolution, app.configs.shadow_resolution)
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
        slnp.lookAt(*(table_pos + np.array([0.6, 0, 1.0])))
        self.world3d.setLight(slnp)

    def on_start(self):
        self.ui_display_region = self.build_kivy_display_region(0, 1.0, 0, 0.061)
        self.kivy_ui = MenuBarUI(
            self.app,
            self.ui_display_region,
            self.id
        )
        self.kivy_ui.run()
    
    def on_destroy(self):
        # !!! important
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim), 
        )

    def forward(self, input):
        return self.mlp(input)


class DeterministicActor(nn.Module):
    def __init__(self, dimo, dimg, dima, hidden_dim=256):
        super().__init__()

        self.trunk = MLP(
            in_dim=dimo+dimg,
            out_dim=dima,
            hidden_dim=hidden_dim
        )

    def forward(self, obs):
        a = self.trunk(obs)
        return torch.tanh(a)

class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.zeros(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
    
    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # do the computing
        self.total_sum += v.sum(axis=0)
        self.total_sumsq += (np.square(v)).sum(axis=0)
        self.total_count[0] += v.shape[0]

    def recompute_stats(self):
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)


class SurgicalSimulator(SurgicalSimulatorBase):
    def __init__(self, env_type, env_params,id=None,demo=None):
        super(SurgicalSimulator, self).__init__(env_type, env_params)
        self.id = id
        if env_type.ACTION_SIZE != 3 and env_type.ACTION_SIZE != 1:
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
            self.app.accept('c-up', self.setPsmAction, [4, 1.0])
            self.app.accept('c-repeat', self.setPsmAction, [4, -0.5])

        self.ecm_view = 0
        self.ecm_view_out = None
        self.demo = demo
        self.start_time = time.time()
        exempt_l = [i for i in range(21,23)]
        if self.id not in exempt_l:
            self.toggleEcmView()
        self.has_load_policy = False

        self.ecm_action = np.zeros(env_type.ACTION_ECM_SIZE)
        if env_type.ACTION_ECM_SIZE == 1:
                self.app.accept('1-up', self.setEcmAction, [0, 0])
                self.app.accept('1-repeat', self.addEcmAction, [0, 0.2])
                self.app.accept('3-up', self.setEcmAction, [0, 0])
                self.app.accept('3-repeat', self.addEcmAction, [0, -0.2])
        else:
            self.app.accept('5-up', self.setEcmAction, [2, 0])
            self.app.accept('5-repeat', self.addEcmAction, [2, 0.2])
            self.app.accept('2-up', self.setEcmAction, [2, 0])
            self.app.accept('2-repeat', self.addEcmAction, [2, -0.2])
            self.app.accept('6-up', self.setEcmAction, [1, 0])
            self.app.accept('6-repeat', self.addEcmAction, [1, 0.2])
            self.app.accept('4-up', self.setEcmAction, [1, 0])
            self.app.accept('4-repeat', self.addEcmAction, [1, -0.2])
            self.app.accept('1-up', self.setEcmAction, [0, 0])
            self.app.accept('1-repeat', self.addEcmAction, [0, 0.2])
            self.app.accept('3-up', self.setEcmAction, [0, 0])
            self.app.accept('3-repeat', self.addEcmAction, [0, -0.2])
            self.app.accept('m-up', self.toggleEcmView)
            self.app.accept('r-up', self.resetEcmFlag)

    def _step_simulation_task(self, task):
        """Step simulation
        """
        if self.demo == None:
            # print(f"scene id:{self.id}")
            if task.time - self.time > 1 / 240.0:
                self.before_simulation_step()

                # Step simulation
                p.stepSimulation()
                self.after_simulation_step()

                # Call trigger update scene (if necessary) and draw methods
                p.getCameraImage(
                    width=1, height=1,
                    viewMatrix=self.env._view_matrix,
                    projectionMatrix=self.env._proj_matrix)
                p.setGravity(0,0,-10.0)
                # print(f"ecm view out matrix:{self.ecm_view_out}")
                self.time = task.time
        else:
            if time.time() - self.time > 1/240:
                self.before_simulation_step()

                # Step simulation
                #pb.stepSimulation()
                # self._duration = 0.1 # needle 
                self._duration = 0.1
                step(self._duration)

                self.after_simulation_step()

                # Call trigger update scene (if necessary) and draw methods
                p.getCameraImage(
                    width=1, height=1,
                    viewMatrix=self.env._view_matrix,
                    projectionMatrix=self.env._proj_matrix)
                p.setGravity(0,0,-10.0)

                self.time = time.time()
                # print(f"current time: {self.time}")
                # print(f"current task time: {task.time}")

                # if time.time()-self.start_time > (self.itr + 1) * time_size:
                obs = self.env._get_obs()
                obs = self.env._get_obs()['achieved_goal'] if isinstance(obs, dict) else None
                # print(f"waypoints: {self.env._waypoints}")
                success = self.env._is_success(obs,self.env._sample_goal()) if obs is not None else False
                print(f"success: {success}")
                wait_list=[12,30]
                if (self.id not in wait_list and success) or time.time()-self.start_time > 10:   
                    # if self.cnt>=6: 
                    #     self.kivy_ui.stop()
                    #     self.app.win.removeDisplayRegion(self.ui_display_region)
                    # self.before_simulation_step()
                    # self._duration = 0.2
                    # step(self._duration)
                    # self.after_simulation_step()
                    
                    open_scene(0)
                    print(f"xxxx current time:{time.time()}")
                    open_scene(self.id)
                    exempt_l = [i for i in range(21,23)]
                    if self.id not in exempt_l:
                        self.toggleEcmView()
                    # self.cnt+=1
                    return 
                    # self.start_time=time.time()
                    # self.toggleEcmView()
                    # self.itr += 1
                        
        return Task.cont

    def load_policy(self, obs, env):
        steps = str(300000)
        if self.id == 8:
            model_file = './peg_transfer_model'
        if self.id == 6:
            model_file = './needle_pick_model'

        actor_params = os.path.join(model_file, 'actor_' + steps + '.pt')
        actor_params = torch.load(actor_params)

        dim_o = obs['observation'].shape[0]
        dim_g = obs['desired_goal'].shape[0]
        dim_a = env.action_space.shape[0]
        actor = DeterministicActor(
            dim_o, dim_g, dim_a, 256
        )

        actor.load_state_dict(actor_params)

        g_norm = Normalizer(dim_g)
        g_norm_stat = np.load(os.path.join(model_file, 'g_norm_' + steps + '_stat.npy'), allow_pickle=True).item()
        g_norm.mean = g_norm_stat['mean']
        g_norm.std = g_norm_stat['std']

        o_norm = Normalizer(dim_o)
        o_norm_stat = np.load(os.path.join(model_file, 'o_norm_' + steps + '_stat.npy'), allow_pickle=True).item()
        o_norm.mean = o_norm_stat['mean']
        o_norm.std = o_norm_stat['std']

        return actor, o_norm, g_norm

    def _preproc_inputs(self, o, g, o_norm, g_norm):
        o_norm = o_norm.normalize(o)
        g_norm = g_norm.normalize(g)
 
        inputs = np.concatenate([o_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        return inputs

    def before_simulation_step(self):
        if (self.id == 8 or self.id == 6) and not self.has_load_policy:
            obs = self.env._get_obs()
            self.actor, self.o_norm,self.g_norm = self.load_policy(obs,self.env)
            self.has_load_policy = True

        if self.demo:
            obs = self.env._get_obs()
            action = self.env.get_oracle_action(obs)

        if self.env.ACTION_SIZE != 3 and self.env.ACTION_SIZE != 1:

            # print(f"len of retrieved action:{len(retrived_action)}")
            if self.demo:
                if self.id ==8:
                    obs = self.env._get_obs()
                    o, g = obs['observation'], obs['desired_goal']
                    input_tensor = self._preproc_inputs(o, g, self.o_norm, self.g_norm)
                    action = self.actor(input_tensor).data.numpy().flatten()
                    print(f"retrieved action is: {self.psm1_action}")
                    self.psm1_action = action
                    self.env._set_action(self.psm1_action)
                else:
                    obs = self.env._get_obs()
                    action = self.env.get_oracle_action(obs)
                    self.psm1_action = action
                    self.env._set_action(self.psm1_action)
                    self.env._step_callback()
            else:
                self.env._set_action(self.psm1_action)
                self.env._step_callback()

        #active track
        if self.id == 27 or self.id == 28:
            self.env._step_callback()

        # self.env._set_action(self.ecm_action)
        if self.demo and (self.env.ACTION_ECM_SIZE == 3 or self.env.ACTION_ECM_SIZE == 1):
            self.ecm_action = action
            self.env._set_action(self.ecm_action)
        if self.demo is None:
            if self.env.ACTION_ECM_SIZE == 1:
                self.env._set_action(self.ecm_action)
            else:
                self.env._set_action_ecm(self.ecm_action)
            
            
        self.env.ecm.render_image()
        # _,_,rgb,_,_ = p.getCameraImage(
        #             width=100, height=100,
        #             viewMatrix=self.env._view_matrix,
        #             projectionMatrix=self.env._proj_matrix
        #             )
        # rgb_array = np.array(rgb, dtype=np.uint8)
        # if len(rgb_array)>0:
        #     rgb_array = np.reshape(rgb_array, (100, 100, 4))

        #     rgb_array = rgb_array[:, :, :3]

        #     imageio.imwrite('test.png',rgb_array)
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
    def resetEcmFlag(self):
        # print("reset enter")
        self.env._reset_ecm_pos()
    def toggleEcmView(self):
        self.ecm_view = not self.ecm_view

    def on_destroy(self):
        # !!! important
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)


class SurgicalSimulatorBimanual(SurgicalSimulatorBase):
    def __init__(self, env_type, env_params, jaw_states=[1.0, 1.0],id=None,demo=None):
        super(SurgicalSimulatorBimanual, self).__init__(env_type, env_params)
        self.id=id
        self.demo = demo
        self.closed=True
        self.start_time = time.time()

        self.psm1_action = np.zeros(env_type.ACTION_SIZE // 2)
        self.psm1_action[4] = jaw_states[0]

        self.psm2_action = np.zeros(env_type.ACTION_SIZE // 2)
        self.psm2_action[4] = jaw_states[1]

        self.app.accept('w-up', self.setPsmAction1, [2, 0])
        self.app.accept('w-repeat', self.addPsmAction1, [2, 0.01])
        self.app.accept('s-up', self.setPsmAction1, [2, 0])
        self.app.accept('s-repeat', self.addPsmAction1, [2, -0.01])
        self.app.accept('d-up', self.setPsmAction1, [1, 0])
        self.app.accept('d-repeat', self.addPsmAction1, [1, 0.01])
        self.app.accept('a-up', self.setPsmAction1, [1, 0])
        self.app.accept('a-repeat', self.addPsmAction1, [1, -0.01])
        self.app.accept('q-up', self.setPsmAction1, [0, 0])
        self.app.accept('q-repeat', self.addPsmAction1, [0, 0.01])
        self.app.accept('e-up', self.setPsmAction1, [0, 0])
        self.app.accept('e-repeat', self.addPsmAction1, [0, -0.01])
        self.app.accept('c-up', self.setPsmAction1, [4, 1.0])
        self.app.accept('c-repeat', self.setPsmAction1, [4, -0.5])

        self.app.accept('i-up', self.setPsmAction2, [2, 0])
        self.app.accept('i-repeat', self.addPsmAction2, [2, 0.01])
        self.app.accept('k-up', self.setPsmAction2, [2, 0])
        self.app.accept('k-repeat', self.addPsmAction2, [2, -0.01])
        self.app.accept('l-up', self.setPsmAction2, [1, 0])
        self.app.accept('l-repeat', self.addPsmAction2, [1, 0.01])
        self.app.accept('j-up', self.setPsmAction2, [1, 0])
        self.app.accept('j-repeat', self.addPsmAction2, [1, -0.01])
        self.app.accept('u-up', self.setPsmAction2, [0, 0])
        self.app.accept('u-repeat', self.addPsmAction2, [0, 0.01])
        self.app.accept('o-up', self.setPsmAction2, [0, 0])
        self.app.accept('o-repeat', self.addPsmAction2, [0, -0.01])
        self.app.accept('n-up', self.setPsmAction2, [4, 1.0])
        self.app.accept('n-repeat', self.setPsmAction2, [4, -0.5])

        self.ecm_view = 0
        self.ecm_view_out = None
        exempt_l = [i for i in range(21,23)]
        if self.id not in exempt_l:
            self.toggleEcmView()
        self.ecm_action = np.zeros(env_type.ACTION_ECM_SIZE)
        self.app.accept('5-up', self.setEcmAction, [2, 0])
        self.app.accept('5-repeat', self.addEcmAction, [2, 0.2])
        self.app.accept('2-up', self.setEcmAction, [2, 0])
        self.app.accept('2-repeat', self.addEcmAction, [2, -0.2])
        self.app.accept('6-up', self.setEcmAction, [1, 0])
        self.app.accept('6-repeat', self.addEcmAction, [1, 0.2])
        self.app.accept('4-up', self.setEcmAction, [1, 0])
        self.app.accept('4-repeat', self.addEcmAction, [1, -0.2])
        self.app.accept('1-up', self.setEcmAction, [0, 0])
        self.app.accept('1-repeat', self.addEcmAction, [0, 0.2])
        self.app.accept('3-up', self.setEcmAction, [0, 0])
        self.app.accept('3-repeat', self.addEcmAction, [0, -0.2])
        self.app.accept('m-up', self.toggleEcmView)
        self.app.accept('r-up', self.resetEcmFlag)

    def _step_simulation_task(self, task):
        """Step simulation
        """
        if self.demo == None:
            # print(f"scene id:{self.id}")
            if task.time - self.time > 1 / 240.0:
                self.before_simulation_step()

                # Step simulation
                p.stepSimulation()
                self.after_simulation_step()

                # Call trigger update scene (if necessary) and draw methods
                p.getCameraImage(
                    width=1, height=1,
                    viewMatrix=self.env._view_matrix,
                    projectionMatrix=self.env._proj_matrix)
                p.setGravity(0,0,-10.0)

                self.time = task.time
        else:
            if time.time() - self.time > 1/240:
                self.before_simulation_step()

                # Step simulation
                #pb.stepSimulation()
                # self._duration = 0.1 # needle 
                self._duration = 0.1
                step(self._duration)

                self.after_simulation_step()

                # Call trigger update scene (if necessary) and draw methods
                p.getCameraImage(
                    width=1, height=1,
                    viewMatrix=self.env._view_matrix,
                    projectionMatrix=self.env._proj_matrix)
                p.setGravity(0,0,-10.0)

                self.time = time.time()
                # print(f"current time: {self.time}")
                # print(f"current task time: {task.time}")

                # if time.time()-self.start_time > (self.itr + 1) * time_size:
                obs = self.env._get_obs()
                obs = self.env._get_obs()['achieved_goal'] if isinstance(obs, dict) else None
                # success = self.env._is_success(obs,self.env._sample_goal()) if obs is not None else False
                if  time.time()-self.start_time > 18:   
                    # if self.cnt>=6: 
                    #     self.kivy_ui.stop()
                    #     self.app.win.removeDisplayRegion(self.ui_display_region)

                    open_scene(0)
                    print(f"xxxx current time:{time.time()}")
                    open_scene(self.id)
                    exempt_l = [i for i in range(21,23)]
                    if self.id not in exempt_l:
                        self.toggleEcmView()
                    # self.cnt+=1
                    return 
                    # self.start_time=time.time()
                    # self.toggleEcmView()
                    # self.itr += 1
                        
        return Task.cont


    def before_simulation_step(self):

        if self.demo:
            obs = self.env._get_obs()
            action = self.env.get_oracle_action(obs)

        if self.demo:
            self.env._set_action(action)
            self.env._step_callback(demo=self.demo)
        else:
            self.env._set_action(np.concatenate([self.psm2_action, self.psm1_action], axis=-1))
            self.env._step_callback()



        # self.env._set_action(self.ecm_action)
        self.env._set_action_ecm(self.ecm_action)
        self.env.ecm.render_image()

        if self.ecm_view:
            self.env._view_matrix = self.env.ecm.view_matrix
        else:
            self.env._view_matrix = self.ecm_view_out

    def setPsmAction1(self, dim, val):
        self.psm1_action[dim] = val
        
    def addPsmAction1(self, dim, incre):
        self.psm1_action[dim] += incre

    def setPsmAction2(self, dim, val):
        self.psm2_action[dim] = val
        
    def addPsmAction2(self, dim, incre):
        self.psm2_action[dim] += incre

    def addEcmAction(self, dim, incre):
        self.ecm_action[dim] += incre

    def setEcmAction(self, dim, val):
        self.ecm_action[dim] = val
    def resetEcmFlag(self):
        # print("reset enter")
        self.env._reset_ecm_pos()
    def toggleEcmView(self):
        self.ecm_view = not self.ecm_view

    def on_destroy(self):
        # !!! important
        self.kivy_ui.stop()
        self.app.win.removeDisplayRegion(self.ui_display_region)


# ecm steoro size 1024x768
app_cfg = ApplicationConfig(window_width=1200, window_height=900)
app = Application(app_cfg)
open_scene(0)
app.run()