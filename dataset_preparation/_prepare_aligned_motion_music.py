import essentia
import essentia.streaming
import glob
import librosa
import numpy as np
import os
import pickle
import sys
sys.path.append('.')
sys.path.append('..')

import torch
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.smpl import SMPLSequence
from copy import deepcopy
from essentia.standard import *
from mutagen.mp3 import MP3
from tools.joint_names import *
from dataset_preparation.dd100_loader import *
from music_features.music_feature_extractor import signal_to_feature
from tools.plot_script import plot_contacts3D
from utils.utils import *


def preprocess_sequence(npy0_path, npy1_path, music_seq_path, save_file_path, seq_data_align_mot_z=False, to_visualize=False):
    
    n_joints = 55
    data0 = np.load(npy0_path, allow_pickle=True, encoding='bytes').item()
    data1 = np.load(npy1_path, allow_pickle=True, encoding='bytes').item()

    if seq_data_align_mot_z:
        data0_preprocessed, _, root_quat_init0, root_pos_init0 = align_mot_z(data0)
        data1_preprocessed, motion1_, root_quat_init1, root_pos_init1 = align_mot_z(data1)
        r_relative = qmul_np(root_quat_init1, qinv_np(root_quat_init0))
        angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])
        xz = qrot_np(root_quat_init0, root_pos_init1 - root_pos_init0)[:, [0, 2]]
        relative = np.concatenate([angle, xz], axis=-1)[0]
        data1_preprocessed, _ = rigid_transform(relative, data1_preprocessed, motion1_)
       
    
    # audio_data, sample_rate = librosa.load(music_seq_path)
    # duration, n_audio_frames, 
    sampling_rate = 15360
    loader = essentia.standard.MonoLoader(filename=music_seq_path, sampleRate=sampling_rate)
    audio = loader()
    audio = np.array(audio).T
    duration = librosa.get_duration(y=audio, sr=sampling_rate)
    music_feature = signal_to_feature(audio, sampling_rate)
    print('mot_seq_length: ', len(data0['transl']))
    print('music_feat_seq_length: ', len(music_feature))

    data_dict = {
        'sequence_name': music_seq_path[:-4],
        'p0_data': data0,
        'p1_data': data1,
        'mot_len': len(data0['transl']),
        'music_feats': music_feature,
        'music_feature_len': len(music_feature),
        'music_duration': duration,
        'music_sampling_rate': sampling_rate,
    }
    # If you want to save the sequences individually
    np.save(makepath(os.path.join(save_file_path, music_seq_path[:-4] + '.npy'), isfile=True), data_dict)

    if to_visualize:
        from tools.visualize_data import visualize_sequence   
        visualize_sequence(data0, data1)   


if __name__ == '__main__':
    # visualize_custom_sequence(os.path.join('data', 'motion', 'smplx', 'Ballet_001_001_preprocessed_.npy'))
    dataset_root = os.path.join('..', 'Datasets', 'DD100') 
    splits = ['train', 'test']
    for split in splits:
        motion_files_dir_path = os.path.join(dataset_root, 'data', 'motion', 'smplx', split)
        music_feature_dir_path = os.path.join(dataset_root, 'data', 'music', 'mp3', split)
        print('Start loading data for split: ', split)
        all_seqs = sorted(glob.glob(music_feature_dir_path + '/*.mp3'))
        print('Completed loading data.')
        
        for music_seq_file_path in all_seqs:
            music_sequence_name = music_seq_file_path.split('/')[-1][:-4]
            p0_seq_path = os.path.join(motion_files_dir_path, music_sequence_name+'_00.npy')
            p1_seq_path = os.path.join(motion_files_dir_path, music_sequence_name+'_01.npy')
            save_file_path = os.path.join('data', 'DD100', split + '_fullsequence')
            preprocess_sequence(p0_seq_path, p1_seq_path, music_seq_file_path, save_file_path, to_visualize=False)