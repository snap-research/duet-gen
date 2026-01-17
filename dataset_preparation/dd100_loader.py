import glob
import os
import pickle
import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from smplx import SMPLX
from dataset_preparation._prepare_data_windows import *
from music_features.music_feature_extractor import FeatureExtractor
from tools.joint_names import *
from utils.transformations import *
from utils.quaternion import *
from utils.utils import *

class DD100_dataset_3(Dataset):
    def __init__(self, data_root=os.path.join('data', 'DD100'), split='train', data_scale=0.1, window_size=128,
                predict_velocity=False,  with_fingers=False, downsample_rate = 1, device='cpu'):
        self.split = split
        self.device = device
        self.data_scale = data_scale
        self.window_size = window_size
        self.with_fingers = with_fingers
        self.predict_velocity = predict_velocity
        self.data_path = os.path.join(data_root, split)
        self.trimmed_music_path = os.path.join(data_root, split.split('_')[0] + '_music', split)
        if with_fingers:
            self.deltamean = to_tensor(np.load(os.path.join(data_root, 'Deltatrans_relMean_full.npy'))).to(device)
            self.deltastd = to_tensor(np.load(os.path.join(data_root, 'Deltatrans_relStd_full.npy'))).to(device)       
        else:
            self.deltamean = to_tensor(np.load(os.path.join(data_root, 'Deltatrans_rel_Mean.npy'))).to(device)
            self.deltastd = to_tensor(np.load(os.path.join(data_root, 'Deltatrans_rel_Std.npy'))).to(device)

        self.genre_list = ['Ballet', 'Foxtrot', 'Jive', 'Lumba', 'PasoDable', 'Qiaqiaqia', 'Quickstep', 'Samba', 'Tango', 'Waltz']
        self.mapping_genre = {}
        for x in range(len(self.genre_list)):
            self.mapping_genre[self.genre_list[x]] = x   
        self.all_seqs = sorted(glob.glob(self.data_path + '/*.pkl')   )
        self.downsample_rate = downsample_rate
        self.data_dict = []
        for seq_idx, sequence in enumerate(tqdm(self.all_seqs)):
            with open(sequence, 'rb') as f:
                data_dict = pickle.load(f)
                data_dict['sequence_name'] = sequence
            if split == 'train' and 'start_time' not in data_dict.keys():
                self.extractor = FeatureExtractor()
                data_dict = self.trim_audio_mel_spectrogram(data_dict, sequence)
                with open(sequence, 'wb') as f:
                    pickle.dump(data_dict, f)
            if split == 'train' and 'p0_globalpos' not in data_dict.keys():
                data_dict = self.calc_jointpos(data_dict)
                with open(sequence, 'wb') as f:
                    pickle.dump(data_dict, f)
            else:
                self.data_dict.append(data_dict)
            
        print("Completed loading " + split + " split. Number of sequences:", len(self.all_seqs) )
        
    
    def z_normalization_delta(self, data):
        return (data - self.deltamean) / (self.deltastd)
    
    def inv_z_normalization_delta(self, data):
        return data * self.deltastd + self.deltamean

    
    def __len__(self):
        return len(self.all_seqs)

    def trim_audio_mel_spectrogram(self, data_dict, sequence):
        if sequence.endswith('_m.pkl'):
            seq_name = sequence[:-6] + '.pkl'
        else:
            seq_name = sequence
        trimmed_music_seq = os.path.join( self.trimmed_music_path, os.path.basename(seq_name))
        with open(trimmed_music_seq, 'rb') as f1:
            music_dict = pickle.load(f1)
        sample_rate = 15360 # used by DD100
        data_dict['start_time'] = music_dict['start_time']
        data_dict['stop_time'] = music_dict['stop_time']
        data_dict['music_path'] = music_dict['music_sequence_name']+'.mp3'
        return data_dict    

    def calc_jointpos(self, data_dict):
        data0_ = mot_to_smplx(data_dict['p0_motion'], betas=data_dict['p0_betas'][0], gender=data_dict['p0_gender'],
                                 with_fingers=True) 
                                
        data_dict['p0_globalpos'] = smplx_to_pos3d(data0_)
        data1_ = mot_to_smplx(data_dict['p1_motion'], betas=data_dict['p1_betas'][0], gender=data_dict['p1_gender'],
                                 with_fingers=True)
        data_dict['p1_globalpos'] = smplx_to_pos3d(data1_)
        return data_dict

    def __getitem__(self, index):
        annot_dict = {}
        data_dict = self.data_dict[index]
        
        annot_dict['p0_betas'] = data_dict['p0_betas'][0:1].to(self.device)
        annot_dict['p1_betas'] = data_dict['p1_betas'][0:1].to(self.device)
        annot_dict['p0_gender'] = 1 if data_dict['p0_gender'] == 'male' else 0
        annot_dict['p1_gender'] = 1 if data_dict['p1_gender'] == 'male' else 0
        annot_dict['music_feats'] = data_dict['music_feats'].to(self.device)
        annot_dict['music_feats_mfcc'] = annot_dict['music_feats'][:, :40]
        annot_dict['music_feats_chroma'] = annot_dict['music_feats'][:, 40:52]
        annot_dict['start_frame'] = data_dict['start_frame']
        annot_dict['stop_frame'] = data_dict['stop_frame']
        annot_dict['dance_genre'] = data_dict['dance_genre']
        annot_dict['genre_class'] = data_dict['genre_class']
        annot_dict['sequence_name'] = data_dict['sequence_name']
        annot_dict['music_start_time'] = data_dict['start_time']
        annot_dict['music_stop_time'] = data_dict['stop_time']
        annot_dict['music_path'] = data_dict['music_path']
        if 'p0_globalpos' in data_dict.keys():
            annot_dict['p0_globalpos'] = data_dict['p0_globalpos'][::self.downsample_rate].to(self.device)
            annot_dict['p1_globalpos'] = data_dict['p1_globalpos'][::self.downsample_rate].to(self.device)
            p0_motion_jts = annot_dict['p0_globalpos'][:, jt_idx_priority]
            p1_motion_jts = annot_dict['p1_globalpos'][:, jt_idx_priority]
            p0_global_pos = torch.repeat_interleave(p0_motion_jts, len(jt_idx_priority), 1)
            p1_global_pos = p1_motion_jts.repeat(1, len(jt_idx_priority), 1)
            relative_dist = (p1_global_pos - p0_global_pos).norm(dim=-1)
        
        if self.with_fingers:
            p0_motion = data_dict['p0_motion'][::self.downsample_rate].to(self.device)
            annot_dict['p0_hand_rot'] = p0_motion[:, REYE_END_INDEX:RIGHT_HAND_ROT_END_INDEX]
            p1_motion = data_dict['p1_motion'][::self.downsample_rate].to(self.device)
            annot_dict['p1_hand_rot'] = p1_motion[:, REYE_END_INDEX:RIGHT_HAND_ROT_END_INDEX]


        else:
            p0_motion = torch.cat((data_dict['p0_motion'][:, :BODYROT_END_INDEX],
                            data_dict['p0_motion'][:, BODYHANDROT_END_INDEX:BODYHANDJOINT2BODYJOINT_END_INDEX],
                            data_dict['p0_motion'][:, BODYHANDJOINT_END_INDEX:BODYHANDJOINTVEL2BODYJOINTVEL_END_INDEX],
                            data_dict['p0_motion'][:, BODYHANDJOINTVEL_END_INDEX:]
            ), dim=-1)[::self.downsample_rate].to(self.device)
            p1_motion = torch.cat((data_dict['p1_motion'][:, :BODYROT_END_INDEX],
                            data_dict['p1_motion'][:, BODYHANDROT_END_INDEX:BODYHANDJOINT2BODYJOINT_END_INDEX],
                            data_dict['p1_motion'][:, BODYHANDJOINT_END_INDEX:BODYHANDJOINTVEL2BODYJOINTVEL_END_INDEX],
                            data_dict['p1_motion'][:, BODYHANDJOINTVEL_END_INDEX:]
            ), dim=-1)[::self.downsample_rate].to(self.device)

        if self.predict_velocity:
            annot_dict['p0_motion_unnormalized'] = p0_motion
            annot_dict['p1_motion_unnormalized'] = p1_motion
            annot_dict['p0_init'] = p0_motion[0:1]
            annot_dict['p1_init'] = p1_motion[0:1]
            p0_root_delta = torch.zeros((len(p0_motion), p0_motion.shape[-1])).to(self.device)
            p1_root_delta = torch.zeros((len(p1_motion), p1_motion.shape[-1])).to(self.device)
            p0_root_delta[1:, :3] = p0_motion[1:, :3] - p0_motion[:-1, :3]
            p0_root_delta[:, 3:] = p0_motion[:, 3:]
            p1_root_delta[:, :3] = p1_motion[:, :3] - p0_motion[:, :3]
            p1_root_delta[:, 3:] = p1_motion[:, 3:] 
            normalized_motion_p0_p1 = self.z_normalization_delta(torch.cat((p0_root_delta, 
                                                                p1_root_delta), dim=-1).to(self.device))


            p0_motion = normalized_motion_p0_p1[..., : normalized_motion_p0_p1.shape[-1]//2].to(self.device)
            p1_motion = normalized_motion_p0_p1[..., normalized_motion_p0_p1.shape[-1]//2 :].to(self.device)
        
        annot_dict['p0_motion'] = p0_motion
        annot_dict['p1_motion'] = p1_motion
        p0_ = p0_motion
        p1_ = p1_motion
        vis=False
        if vis:
            if self.predict_velocity:
                unnormalized_mot = self.inv_z_normalization_delta(torch.cat((annot_dict['p0_motion'], 
                                                                  annot_dict['p1_motion']), dim=-1).to(self.device))
                p0_delta_ = unnormalized_mot[:, : unnormalized_mot.shape[-1]//2]
                p1_delta_ = unnormalized_mot[:, unnormalized_mot.shape[-1]//2:]
                p0_, p1_ = delta_rel_trans_mot2mot_feats(annot_dict['p0_init'].unsqueeze(0), p0_delta_.unsqueeze(0), p1_delta_.unsqueeze(0))
                p0_ = p0_.squeeze(0)
                p1_ = p1_.squeeze(0) 
            data0 = mot_to_smplx(p0_, betas=annot_dict['p0_betas'], gender='female' if annot_dict['p0_gender'] == 0 else 'male',
                                with_fingers=self.with_fingers)            
            data1 = mot_to_smplx(p1_, betas=annot_dict['p1_betas'], gender='female' if annot_dict['p1_gender'] == 0 else 'male',
                                with_fingers=self.with_fingers) 
            data0_ = mot_to_smplx(data_dict['p0_motion'][::self.downsample_rate], betas=annot_dict['p0_betas'], gender='female' if annot_dict['p0_gender'] == 0 else 'male',
                                 with_fingers=True)                      
            data1_ = mot_to_smplx(data_dict['p1_motion'][::self.downsample_rate], betas=annot_dict['p1_betas'], gender='female' if annot_dict['p1_gender'] == 0 else 'male',
                                 with_fingers=True) 
            from tools.visualize_data import visualize_sequence
            visualize_sequence(data0, data1, data0_, data1_)
        return annot_dict