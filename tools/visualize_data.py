import ffmpeg
import glob
import numpy as np
import os
import sys
sys.path.append('.')
sys.path.append('..')
import skvideo
# skvideo.setFFmpegPath('/opt/homebrew/bin')
# MESA_GL_VERSION_OVERRIDE=4.0
import skvideo.io
import torch
import trimesh
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from copy import deepcopy
from dataset_preparation.data_utils import *
from smplx import SMPLX
from tools.joint_names import *
from tools.plot_script import plot_contacts3D
from utils.transformations import *
from utils.quaternion import *
from utils.utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


'''Add the following snippet to the SMPLSequence class in the aitviewer.renderables.smpl

@classmethod
    def from_custom_npy(cls, data, z_up=True, **kwargs):
        smpl_layer = SMPLLayer(model_type="smplx", gender=data["meta"]["gender"], device=C.device)
        poses = data["poses"]
        trans = data["transl"]
        i_root_end = 3
        i_body_end = i_root_end + smpl_layer.bm.NUM_BODY_JOINTS * 3
        i_left_hand_end = i_body_end + smpl_layer.bm.NUM_HAND_JOINTS * 3
        i_right_hand_end = i_left_hand_end + smpl_layer.bm.NUM_HAND_JOINTS * 3
        return cls(
            poses_body=poses[:, i_root_end:i_body_end],
            poses_root=poses[:, :i_root_end],
            poses_left_hand=poses[:, i_body_end:i_left_hand_end],
            poses_right_hand=poses[:, i_left_hand_end:i_right_hand_end],
            smpl_layer=smpl_layer,
            betas=data["betas"],
            trans=trans,
            z_up=z_up,
            **kwargs,
        )

'''


def visualize_sequence(data0, data1=None, data2=None, data3=None):
    v = Viewer()
    v.scene.add(SMPLSequence.from_custom_npy(data=data0, z_up=True, color=(0.6, 0.4, 0.4, 1.0)))
    if data1 is not None:
        v.scene.add(SMPLSequence.from_custom_npy(data=data1, z_up=True, color=(0.5, 0.5, 0.8, 1.0)))
    if data2 is not None:
        v.scene.add(SMPLSequence.from_custom_npy(data=data2, z_up=True, color=(0.6, 0.4, 0.4, 1.0)))
    if data3 is not None:
        v.scene.add(SMPLSequence.from_custom_npy(data=data3, z_up=True, color=(0.5, 0.5, 0.8, 1.0)))
    v.run()

def save_sequence(v, save_path, data0, data1=None, data2=None, data3=None):
    
    v.reset()
    smpl0 = SMPLSequence.from_custom_npy(data=data0, z_up=True, color=(0.6, 0.4, 0.4, 1.0))
    v.scene.add(smpl0)
    if data1 is not None:
        v.scene.add(SMPLSequence.from_custom_npy(data=data1, z_up=True, color=(0.5, 0.5, 0.8, 1.0)))
    if data2 is not None:
        v.scene.add(SMPLSequence.from_custom_npy(data=data2, z_up=True, color=(0.6, 0.4, 0.4, 1.0)))
    if data3 is not None:
        v.scene.add(SMPLSequence.from_custom_npy(data=data3, z_up=True, color=(0.5, 0.5, 0.8, 1.0)))
    # v.scene.camera
    # v.auto_set_camera_target = False
    v.scene.camera.position = np.array((-5.0, 1.5, 0))
    v.scene.camera.target = np.array((0.0, 1.0, 0.0))
    # v.lock_to_node(smpl0, (3.5, 3.5, 3.5), smooth_sigma=5.0)
    v.save_video(video_dir=save_path, output_fps=30, ensure_no_overwrite=False) #, rotate_camera=False, rotation_degrees=360.0)
    # v.save_video(video_dir=save_path, output_fps=30, rotate_camera=False, rotation_degrees=360.0)
    print("video saved at ", save_path)


def combine_with_audio(save_path, music_path, music_start, music_end):
    save_with_music_path = save_path[:-4]+'_music.mp4'
    out = ffmpeg.output(ffmpeg.input(save_path[:-4]+'_0.mp4'), ffmpeg.input(music_path), save_with_music_path, vcodec='copy', acodec='aac', strict='experimental')
    out.run()
    os.remove(save_path[:-4]+'_0.mp4')


def save_dataset_sequence(v, data0, save_path, data1=None,):
    v.reset()
    smpl0 = SMPLSequence.from_custom_npy(data=data0, z_up=True, color=(0.6, 0.5, 0.5, 1.0))
    v.scene.add(smpl0)
    if data1 is not None:
        smpl1 = SMPLSequence.from_custom_npy(data=data1, z_up=True, color=(0.5, 0.5, 0.6, 1.0))
        v.scene.add(smpl1)
    # v.scene.camera
    v.auto_set_camera_target = False
    v.scene.camera.position = np.array((11.0, 4.0, -9))
    v.scene.camera.target = np.array((0.0, 0.6, 0.0))
    v.lock_to_node(smpl0, (4, 4, 4), smooth_sigma=5.0)
    v.save_video(video_dir=save_path, output_fps=30, rotate_camera=False, rotation_degrees=360.0)
    print("video saved at ", save_path)
