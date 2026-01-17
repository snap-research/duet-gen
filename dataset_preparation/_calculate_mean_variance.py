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

def delta_mean_variance_DD100_dataset(all_seqs, with_fingers=True):
    data_list = []
    for idx, filename in enumerate(all_seqs):
        with open(filename, 'rb') as f:
            data_dict = np.load(f, allow_pickle=True)
        if with_fingers:
            p0_motion = data_dict['p0_motion']
            p1_motion = data_dict['p1_motion']
        else:
            p0_motion = torch.cat((data_dict['p0_motion'][:, :BODYROT_END_INDEX],
                               data_dict['p0_motion'][:, BODYHANDROT_END_INDEX:BODYHANDJOINT2BODYJOINT_END_INDEX],
                               data_dict['p0_motion'][:, BODYHANDJOINT_END_INDEX:BODYHANDJOINTVEL2BODYJOINTVEL_END_INDEX],
                               data_dict['p0_motion'][:, BODYHANDJOINTVEL_END_INDEX:]
            ), dim=-1)
            p1_motion = torch.cat((data_dict['p1_motion'][:, :BODYROT_END_INDEX],
                               data_dict['p1_motion'][:, BODYHANDROT_END_INDEX:BODYHANDJOINT2BODYJOINT_END_INDEX],
                               data_dict['p1_motion'][:, BODYHANDJOINT_END_INDEX:BODYHANDJOINTVEL2BODYJOINTVEL_END_INDEX],
                               data_dict['p1_motion'][:, BODYHANDJOINTVEL_END_INDEX:]
            ), dim=-1)                  
        with torch.no_grad():
            p0_root_delta = torch.zeros((len(p0_motion), p0_motion.shape[-1]))
            p1_root_delta = torch.zeros((len(p1_motion), p1_motion.shape[-1]))
            p0_root_delta[1:, :9] = p0_motion[1:, :9] - p0_motion[:-1, :9]
            p0_root_delta[:, 9:] = p0_motion[:, 9:].clone()
            p1_root_delta[1:, :9] = p1_motion[1:, :9] - p1_motion[:-1, :9]
            p1_root_delta[:, 9:] = p1_motion[:, 9:].clone()
        data = to_np(torch.cat((p0_root_delta, p1_root_delta), dim=-1))
        if np.isnan(data).any():
            print(filename + "doesnot exist")
            continue
        data_list.append(data)
        print(idx)

    if with_fingers:
        mean_save_path = os.path.join(data_dir, 'DeltaMean_full.npy')
        std_save_path = os.path.join(data_dir, 'DeltaStd_full.npy')
    else:
        mean_save_path = os.path.join(data_dir, 'DeltaMean.npy')
        std_save_path = os.path.join(data_dir, 'DeltaStd.npy')
    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    print(Mean.shape)
    np.save(mean_save_path, Mean)
    Std = data.std(axis=0) + 1e-8
    print(Std.shape)
    np.save(std_save_path, Std)


def delta_trans_mean_variance_DD100_dataset(all_seqs, with_fingers=False):
    data_list = []
    for idx, filename in enumerate(all_seqs):
        with open(filename, 'rb') as f:
            data_dict = np.load(f, allow_pickle=True)
        if with_fingers:
            p0_motion = data_dict['p0_motion']
            p1_motion = data_dict['p1_motion']
        else:
            p0_motion = torch.cat((data_dict['p0_motion'][:, :BODYROT_END_INDEX],
                               data_dict['p0_motion'][:, BODYHANDROT_END_INDEX:BODYHANDJOINT2BODYJOINT_END_INDEX],
                               data_dict['p0_motion'][:, BODYHANDJOINT_END_INDEX:BODYHANDJOINTVEL2BODYJOINTVEL_END_INDEX],
                               data_dict['p0_motion'][:, BODYHANDJOINTVEL_END_INDEX:]
            ), dim=-1)
            p1_motion = torch.cat((data_dict['p1_motion'][:, :BODYROT_END_INDEX],
                               data_dict['p1_motion'][:, BODYHANDROT_END_INDEX:BODYHANDJOINT2BODYJOINT_END_INDEX],
                               data_dict['p1_motion'][:, BODYHANDJOINT_END_INDEX:BODYHANDJOINTVEL2BODYJOINTVEL_END_INDEX],
                               data_dict['p1_motion'][:, BODYHANDJOINTVEL_END_INDEX:]
            ), dim=-1)                  
        with torch.no_grad():
            p0_root_delta = torch.zeros((len(p0_motion), p0_motion.shape[-1]))
            p1_root_delta = torch.zeros((len(p1_motion), p1_motion.shape[-1]))
            p0_root_delta[1:, :3] = p0_motion[1:, :3] - p0_motion[:-1, :3]
            p0_root_delta[:, 3:] = p0_motion[:, 3:].clone()
            p1_root_delta[1:, :3] = p1_motion[1:, :3] - p1_motion[:-1, :3]
            p1_root_delta[:, 3:] = p1_motion[:, 3:].clone()
        data = to_np(torch.cat((p0_root_delta, p1_root_delta), dim=-1))
        if np.isnan(data).any():
            print(filename + "doesnot exist")
            continue
        data_list.append(data)
        print(idx)

    if with_fingers:
        mean_save_path = os.path.join(data_dir, 'DeltaMean_full.npy')
        std_save_path = os.path.join(data_dir, 'DeltaStd_full.npy')
    else:
        mean_save_path = os.path.join(data_dir, 'Delta_transMean.npy')
        std_save_path = os.path.join(data_dir, 'Delta_transStd.npy')
    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    print(Mean.shape)
    np.save(mean_save_path, Mean)
    Std = data.std(axis=0) + 1e-8
    print(Std.shape)
    np.save(std_save_path, Std)



def delta_first_trans_mean_variance_DD100_dataset(all_seqs, with_fingers=False):
    data_list = []
    for idx, filename in enumerate(all_seqs):
        with open(filename, 'rb') as f:
            data_dict = np.load(f, allow_pickle=True)
        if with_fingers:
            p0_motion = data_dict['p0_motion']
            p1_motion = data_dict['p1_motion']
        else:
            p0_motion = torch.cat((data_dict['p0_motion'][:, :BODYROT_END_INDEX],
                               data_dict['p0_motion'][:, BODYHANDROT_END_INDEX:BODYHANDJOINT2BODYJOINT_END_INDEX],
                               data_dict['p0_motion'][:, BODYHANDJOINT_END_INDEX:BODYHANDJOINTVEL2BODYJOINTVEL_END_INDEX],
                               data_dict['p0_motion'][:, BODYHANDJOINTVEL_END_INDEX:]
            ), dim=-1)
            p1_motion = torch.cat((data_dict['p1_motion'][:, :BODYROT_END_INDEX],
                               data_dict['p1_motion'][:, BODYHANDROT_END_INDEX:BODYHANDJOINT2BODYJOINT_END_INDEX],
                               data_dict['p1_motion'][:, BODYHANDJOINT_END_INDEX:BODYHANDJOINTVEL2BODYJOINTVEL_END_INDEX],
                               data_dict['p1_motion'][:, BODYHANDJOINTVEL_END_INDEX:]
            ), dim=-1)                  
        with torch.no_grad():
            p0_root_delta = torch.zeros((len(p0_motion), p0_motion.shape[-1]))
            p1_root_delta = torch.zeros((len(p1_motion), p1_motion.shape[-1]))
            p0_root_delta[:, :3] = p0_motion[:, :3] - p0_motion[0, :3]
            p0_root_delta[:, 3:] = p0_motion[:, 3:].clone()
            p1_root_delta[:, :3] = p1_motion[:, :3] - p0_motion[:, :3]
            p1_root_delta[:, 3:] = p1_motion[:, 3:].clone()
        data = to_np(torch.cat((p0_root_delta, p1_root_delta), dim=-1))
        if np.isnan(data).any():
            print(filename + "doesnot exist")
            continue
        data_list.append(data)
        print(idx)

    if with_fingers:
        mean_save_path = os.path.join(data_dir, 'DeltafirstMean_full.npy')
        std_save_path = os.path.join(data_dir, 'DeltafirstStd_full.npy')
        data_save_path = os.path.join(data_dir, 'Deltafirst_full.npy')
    else:
        mean_save_path = os.path.join(data_dir, 'Deltafirst_transMean.npy')
        std_save_path = os.path.join(data_dir, 'Deltafirst_transStd.npy')
        data_save_path = os.path.join(data_dir, 'Deltafirst_trans.npy')
    data = np.concatenate(data_list, axis=0)
    np.save(data_save_path, data)
    print(data.shape)
    Mean = data.mean(axis=0)
    print(Mean.shape)
    np.save(mean_save_path, Mean)
    Std = data.std(axis=0) + 1e-8
    print(Std.shape)
    np.save(std_save_path, Std)

def delta_rel_trans_mean_variance_DD100_dataset(all_seqs, with_fingers=False):
    data_list = []
    for idx, filename in enumerate(all_seqs):
        with open(filename, 'rb') as f:
            data_dict = np.load(f, allow_pickle=True)
        if with_fingers:
            p0_motion = data_dict['p0_motion']
            p1_motion = data_dict['p1_motion']
        else:
            p0_motion = torch.cat((data_dict['p0_motion'][:, :BODYROT_END_INDEX],
                               data_dict['p0_motion'][:, BODYHANDROT_END_INDEX:BODYHANDJOINT2BODYJOINT_END_INDEX],
                               data_dict['p0_motion'][:, BODYHANDJOINT_END_INDEX:BODYHANDJOINTVEL2BODYJOINTVEL_END_INDEX],
                               data_dict['p0_motion'][:, BODYHANDJOINTVEL_END_INDEX:]
            ), dim=-1)
            p1_motion = torch.cat((data_dict['p1_motion'][:, :BODYROT_END_INDEX],
                               data_dict['p1_motion'][:, BODYHANDROT_END_INDEX:BODYHANDJOINT2BODYJOINT_END_INDEX],
                               data_dict['p1_motion'][:, BODYHANDJOINT_END_INDEX:BODYHANDJOINTVEL2BODYJOINTVEL_END_INDEX],
                               data_dict['p1_motion'][:, BODYHANDJOINTVEL_END_INDEX:]
            ), dim=-1)                  
        with torch.no_grad():
            p0_root_delta = torch.zeros((len(p0_motion), p0_motion.shape[-1]))
            p1_root_delta = torch.zeros((len(p1_motion), p1_motion.shape[-1]))
            p0_root_delta[1:, :3] = p0_motion[1:, :3] - p0_motion[:-1, :3]
            p0_root_delta[:, 3:] = p0_motion[:, 3:].clone()
            p1_root_delta[:, :3] = p1_motion[:, :3] - p0_motion[:, :3]
            p1_root_delta[:, 3:] = p1_motion[:, 3:].clone()
        data = to_np(torch.cat((p0_root_delta, p1_root_delta), dim=-1))
        if np.isnan(data).any():
            print(filename + "doesnot exist")
            continue
        data_list.append(data)
        print(idx)

    if with_fingers:
        mean_save_path = os.path.join(data_dir, 'Deltatrans_relMean_full.npy')
        std_save_path = os.path.join(data_dir, 'Deltatrans_relStd_full.npy')
        data_save_path = os.path.join(data_dir, 'Deltatrans_rel_full.npy')
    else:
        mean_save_path = os.path.join(data_dir, 'Deltatrans_rel_Mean.npy')
        std_save_path = os.path.join(data_dir, 'Deltatrans_rel_Std.npy')
        data_save_path = os.path.join(data_dir, 'Deltatrans_rel.npy')
    data = np.concatenate(data_list, axis=0)
    np.save(data_save_path, data)
    print(data.shape)
    Mean = data.mean(axis=0)
    print(Mean.shape)
    np.save(mean_save_path, Mean)
    Std = data.std(axis=0) + 1e-8
    print(Std.shape)
    np.save(std_save_path, Std)

if __name__ == '__main__':
    splits = ['train', 'test']
    window_size = [128, 400]
    window_stride = [32, 100]
    for split in splits:
        for i in range(len(window_size)):
            data_dir = os.path.join('data', 'DD100', str(window_size[i]) + '_' + str(window_stride[i]))
            all_seqs = glob.glob(data_dir + '/*/*.pkl')
            delta_rel_trans_mean_variance_DD100_dataset(all_seqs, with_fingers=False)
    