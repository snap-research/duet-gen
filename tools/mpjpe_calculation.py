import numpy as np
import os
import pickle
import sys
sys.path.append('.')
sys.path.append('..')
import torch
from dataset_preparation.data_utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def load_pkl_files_and_calculate_mean(file_path, num_samples):

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
        # Extract the values from the dictionary and extend them to the list
        # Assuming the values are numerical
        num_samples = num_samples/1000
        p0_mpjgpe = (data['p0_mpjgpe'] / num_samples).mean()
        p1_mpjgpe = (data['p1_mpjgpe'] / num_samples).mean()
        p0_mpjrpe = (data['p0_mpjrpe'] / num_samples).mean()
        p1_mpjrpe = (data['p1_mpjrpe'] / num_samples).mean()
        p0_mroot_pe = (data['p0_mroot_pe'] / num_samples).mean()
        p1_mroot_pe = (data['p1_mroot_pe'] / num_samples).mean()
        p0_mpjve = (data['p0_mpjve'] / num_samples).mean()
        p1_mpjve = (data['p1_mpjve'] / num_samples).mean()
        p0_mroot_re = (data['p0_mroot_re'] / num_samples).mean()
        p1_mroot_re = (data['p1_mroot_re'] / num_samples).mean()
        print(' Mean per joint global position error - P0: {:.5f} mm'.format(p0_mpjgpe))
        print(' Mean per joint global position error - P1: {:.5f} mm'.format(p1_mpjgpe))
        print(' Mean per joint relative position error - P0: {:.5f} mm'.format(p0_mpjrpe))
        print(' Mean per joint relative position error - P1: {:.5f} mm'.format(p1_mpjrpe))
        print(' Mean root position error - P0: {:.5f} mm'.format(p0_mroot_pe))
        print(' Mean root position error - P1: {:.5f} mm'.format(p1_mroot_pe))
        print(' Mean root orientation error - P0: {:.5f} mm'.format(p0_mroot_re))
        print(' Mean root orientation error - P1: {:.5f} mm'.format(p1_mroot_re))

        print(' Mean per joint velocity error - P0: {:.5f} mm'.format(p0_mpjve))
        print(' Mean per joint velocity error - P1: {:.5f} mm'.format(p1_mpjve))
        
    

    
if __name__ == "__main__":
    # Example usage
    num_samples = 1100
    directory_path = os.path.join('checkpoints', 'DD100', 'vqvae', 'exp_5_Hierarchical_VQVAE_TwoPerson_64_400_nofingers', '4175', 'MPJPE_', str(num_samples) +'.pkl')
    # directory_path = os.path.join('checkpoints', 'DD100', 'vqvae', 'exp_103_MotionVQVAE_Music_256_128_nofingers', '1225', 'MPJPE_', str(num_samples) +'.pkl')
    load_pkl_files_and_calculate_mean(directory_path, num_samples)
    print("end")
