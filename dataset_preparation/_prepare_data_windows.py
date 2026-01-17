import glob
import os
import pickle
import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')
import torch
from copy import deepcopy
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from smplx import SMPLX
from dataset_preparation.data_utils import *
from tools.joint_names import *
from utils.transformations import *
from utils.quaternion import *
from utils.utils import *



def extract_data_for_frames(p0_data, p1_data=None, music_feats=None, start_frame=0, stop_frame=100, skip=1):
    if music_feats is not None:
        music_short = to_tensor(music_feats[start_frame:stop_frame], device=device)
    else: music_short = None
    p0_short = {}
    p0_short['transl'] = to_tensor(p0_data['transl'][start_frame:stop_frame:skip], device=device)
    p0_short['poses'] = to_tensor(p0_data['poses'][start_frame:stop_frame:skip], device=device)
    p0_short['global_orient'] = to_tensor(p0_data['global_orient'][start_frame:stop_frame:skip], device=device)
    p0_short['betas'] = to_tensor(p0_data['betas'][start_frame:stop_frame:skip, :10], device=device)
    p0_short['meta'] = {'gender': p0_data['meta']['gender'] }

    if p1_data is not None:
        p1_short = {}
        p1_short['transl'] = to_tensor(p1_data['transl'][start_frame:stop_frame:skip], device=device)
        p1_short['poses'] = to_tensor(p1_data['poses'][start_frame:stop_frame:skip], device=device)
        p1_short['global_orient'] = to_tensor(p1_data['global_orient'][start_frame:stop_frame:skip], device=device)
        p1_short['betas'] = to_tensor(p1_data['betas'][start_frame:stop_frame:skip, :10], device=device)
        p1_short['meta'] = {'gender': p1_data['meta']['gender'] }
    else: p1_short = None
    return p0_short, p1_short, music_short





def sequence_data_align_mot_z_and_transl(data0, data1=None, feet_thre=0.002, with_fingers=True):
    data0_preprocessed, _, root_quat_init0, root_pos_init0, data0_rootjoint_offset = align_mot_z(data0)
    motion0 = new_process_mot(data0_preprocessed, feet_thre=feet_thre, with_fingers=with_fingers)
    if data1 is not None:
        data1_preprocessed, motion1_, root_quat_init1, root_pos_init1, data1_rootjoint_offset = align_mot_z(data1)
        r_relative = qmul(root_quat_init1, qinv(root_quat_init0))
        angle = torch.arctan2(r_relative[2:3], r_relative[0:1])
        xz = qrot(root_quat_init0, root_pos_init1 - root_pos_init0)[[0, 2]]
        relative = torch.cat([angle, xz], dim=-1)
        data1_preprocessed, _ = rigid_transform(relative, data1_preprocessed, motion1_)
        motion1 = new_process_mot(data1_preprocessed, feet_thre=feet_thre, with_fingers=with_fingers)
    else: 
        data1_preprocessed = None
        motion1 = None
    return data0_preprocessed, data1_preprocessed, motion0, motion1, relative, data0_rootjoint_offset, data1_rootjoint_offset


def one_hot_vectors(word, word_list, mapping):
    arr = list(np.zeros(len(word_list), dtype = int))
    arr[mapping[word]] = 1
    return np.array(arr)

class preprocess_DD100_dataset():
    def __init__(self, data_root=os.path.join('data', 'DD100'), split='train', 
                 window_size=400, window_stride=100, with_fingers=True, mirror=False):
        self.split = split
        self.window_size = window_size
        self.window_stride = window_stride
        self.mirror = mirror
        self.with_fingers = with_fingers
        self.data_root = data_root
        self.data_path = os.path.join(data_root, split + '_fullsequence')
        self.genre_list = ['Ballet', 'Foxtrot', 'Jive', 'Lumba', 'PasoDable', 'Qiaqiaqia', 'Quickstep', 'Samba', 'Tango', 'Waltz']
        self.mapping_genre = {}
        for x in range(len(self.genre_list)):
            self.mapping_genre[self.genre_list[x]] = x   
        self.all_seqs = sorted(glob.glob(self.data_path + '/*.npy')   )
        len_data_dict = len(self.all_seqs)
        print("Completed loading " + split + " split")
        for seq_idx in range(0, len_data_dict):
            self.getitem_(seq_idx)
            self.getitem_music(seq_idx)
            print("completed sequence: ", seq_idx)
            
        
    
    def getitem_music(self, index):     
        index = index % len(self.all_seqs)
        data_dict_path = self.all_seqs[index]
        with open(data_dict_path, 'rb') as f:
            data_dict = np.load(f, allow_pickle=True)
        data_dict = dict(enumerate(data_dict.flatten(), 0))[0]
        dance_genre = data_dict['sequence_name'].split('/')[-1].split('_')[0]
        music_frame_rate = data_dict['music_feature_len'] / data_dict['music_duration']
        seq_len = data_dict['mot_len'] if data_dict['mot_len'] < data_dict['music_feature_len'] else data_dict['music_feature_len']
        wind_indx = 0
        for start_frame in range(0, seq_len - self.window_size,  self.window_stride):
            stop_frame = start_frame + self.window_size
            music_data = {
                 'index': index,
                 'music_sequence_name': data_dict['sequence_name'],
                 'start_time': start_frame/music_frame_rate, 
                 'stop_time': stop_frame/music_frame_rate
            }

            save_file_path = makepath(os.path.join(self.data_root, str(self.window_size) + '_' + str(self.window_stride),
                                                    split+'_music', str(index) + '_' + str(wind_indx) +  '.pkl'), isfile=True)
            with open(save_file_path, 'wb') as f:
                pickle.dump(music_data, f)
            wind_indx += 1
            print("subsequence: ", wind_indx)
             
    def getitem_count(self, index):     
        index = index % len(self.all_seqs)
        data_dict_path = self.all_seqs[index]
        with open(data_dict_path, 'rb') as f:
            data_dict = np.load(f, allow_pickle=True)
        data_dict = dict(enumerate(data_dict.flatten(), 0))[0]
        dance_genre = data_dict['sequence_name'].split('/')[-1].split('_')[0]
        print(dance_genre)
                 
    def getitem_(self, index):     
        index = index % len(self.all_seqs)
        data_dict_path = self.all_seqs[index]
        with open(data_dict_path, 'rb') as f:
            data_dict = np.load(f, allow_pickle=True)
        data_dict = dict(enumerate(data_dict.flatten(), 0))[0]
        dance_genre = data_dict['sequence_name'].split('/')[-1].split('_')[0]
        seq_len = data_dict['mot_len'] if data_dict['mot_len'] < data_dict['music_feature_len'] else data_dict['music_feature_len']
        wind_indx = 0
        for start_frame in range(0, seq_len - self.window_size,  self.window_stride):
            annot_dict = {
                'index': [],
                'p0_smplx': [],
                'p1_smplx': [],
                'p0_motion': [],
                'p1_motion': [],
                'relative_pos': [],
                'relative_rot': [],
                'p0_betas': [],
                'p1_betas': [],
                'p0_gender': [],
                'p1_gender': [],
                'music_feats': [],
                'dance_genre': [],
                'genre_class': [],
                'start_frame': [],
                'stop_frame': [],
                'relative': [],
            }
            # start_frame = np.random.randint(0, seq_len - self.window_size)
            stop_frame = start_frame + self.window_size
            p0_data, p1_data, music_feats = extract_data_for_frames(data_dict['p0_data'], data_dict['p1_data'],
                                                                    data_dict['music_feats'], start_frame=start_frame, stop_frame=stop_frame )
            
            annot_dict['index'] = index
            annot_dict['music_feats'] = music_feats
            annot_dict['start_frame'] = start_frame
            annot_dict['stop_frame'] = stop_frame
            annot_dict['dance_genre'] = dance_genre
            annot_dict['genre_class'] = one_hot_vectors(dance_genre, self.genre_list, self.mapping_genre)
            
            
            p0_aligned_smplx, p1_aligned_smplx, p0_motion, p1_motion, relative, _,_ = sequence_data_align_mot_z_and_transl(copy(p0_data), 
                                                                                                                                                            copy(p1_data), 
                                                                                                                                                            feet_thre=0.001,
                                                                                                                                                            with_fingers=self.with_fingers)
            annot_dict['p0_motion'] = p0_motion
            annot_dict['p1_motion'] = p1_motion
            annot_dict['relative_pos'] = p1_motion[:, :3] - p0_motion[:, :3] 
            annot_dict['relative_rot'] = p1_motion[:, 3:9] - p0_motion[:, 3:9] 
            annot_dict['p0_betas'] = p0_aligned_smplx['betas'][0:1]
            annot_dict['p1_betas'] = p1_aligned_smplx['betas'][0:1]
            annot_dict['p0_gender'] = p0_aligned_smplx['meta']['gender']
            annot_dict['p1_gender'] = p1_aligned_smplx['meta']['gender']
            annot_dict['relative'] = relative



            save_file_path = makepath(os.path.join(self.data_root, str(self.window_size) + '_' + str(self.window_stride),
                                                    split, str(index) + '_' + str(wind_indx) +  '.pkl'), isfile=True)
            vis=False
            if vis:
                data0 = mot_to_smplx(annot_dict['p0_motion'], betas=annot_dict['p0_betas'], gender= annot_dict['p0_gender']) 
                                    #  root_offset=annot_dict['data0_rootjoint_offset'])
                data1 = mot_to_smplx(annot_dict['p1_motion'], betas=annot_dict['p1_betas'], gender= annot_dict['p1_gender'])
                                    #  root_offset=annot_dict['data1_rootjoint_offset'])
                from tools.visualize_data import visualize_sequence
                visualize_sequence(data0, data1) #, p0_data, p1_data)
            with open(save_file_path, 'wb') as f:
                pickle.dump(annot_dict, f)
            if self.mirror:
                p0_aligned_smplx, p1_aligned_smplx, p0_motion, p1_motion, relative, _, _ = sequence_data_align_mot_z_and_transl(p1_data, p0_data, 
                                                                                        feet_thre=0.001, with_fingers=self.with_fingers)
                annot_dict['p0_motion'] = p0_motion
                annot_dict['p1_motion'] = p1_motion
                annot_dict['relative_pos'] = p1_motion[:, :3] - p0_motion[:, :3] 
                annot_dict['relative_rot'] = p1_motion[:, 3:9] - p0_motion[:, 3:9] 
                annot_dict['p0_betas'] = p0_aligned_smplx['betas']
                annot_dict['p1_betas'] = p1_aligned_smplx['betas']
                annot_dict['p0_gender'] = p0_aligned_smplx['meta']['gender']
                annot_dict['p1_gender'] = p1_aligned_smplx['meta']['gender']
                annot_dict['relative'] = relative
                if vis:
                    data0 = mot_to_smplx(annot_dict['p0_motion'], betas=annot_dict['p0_betas'], gender= annot_dict['p0_gender']) 
                                        #  root_offset=annot_dict['data0_rootjoint_offset'])
                    data1 = mot_to_smplx(annot_dict['p1_motion'], betas=annot_dict['p1_betas'], gender= annot_dict['p1_gender'])
                                        #  root_offset=annot_dict['data1_rootjoint_offset'])
                    from tools.visualize_data import visualize_sequence
                    visualize_sequence(data0, data1, p0_data, p1_data)
                save_file_path = makepath(os.path.join(self.data_root,
                                                       str(self.window_size) + '_' + str(self.window_stride), 
                                                       split, str(index) + '_' + str(wind_indx) +  '_m.pkl'), isfile=True)
                with open(save_file_path, 'wb') as f:
                    pickle.dump(annot_dict, f)
            wind_indx += 1
            print("subsequence: ", wind_indx)
            

if __name__ == '__main__':

    splits = ['train', 'test']
    window_size = [128, 400]
    window_stride = [32, 100]
    for split in splits:
        for i in range(len(window_size)):
            preprocess_DD100_dataset(split=split, window_size=window_size[i],
                            window_stride=window_stride[i], with_fingers=True, mirror=True)
    
    
 
 

 
