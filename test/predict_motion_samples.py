import os
import numpy as np
import pickle
import sys
sys.path.append('.')
sys.path.append('..')
import time
import torch
import tqdm
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.transformations import *
from dataset_preparation.dd100_loader import DD100_dataset_3 as DD100_dataset
from dataset_preparation.data_utils import *
from models.vqvae.motion_vqvae import *
from models.extra_motion_pred import *
from models.transformer.music2motion import *
from utils.metrics import *
from train.args_masked_transformer_bottom import arg_parse
from utils.utils import *

test_split = 'test'  # the dataset split that you want to evaluate on.
load_exp = os.path.join('checkpoints', 'DD100', 'masked_transformer', 'exp_18_Music2Motion_combined_hier_bottom_128_400_nofingers', '1675', 'weights.p')
traj_pred_model_pretrained_weight_path = os.path.join('checkpoints', 'DD100', 'extra_motion', 'exp_16_Global_Trajectory_Pred_256_128_nofingers', '5501', 'weights.p')

class MaskedTransformerTrainer:
    def __init__(self, args):
        self.opt = args
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print('Using ', self.opt.gpu_id, ' GPUs!')
        
        if self.opt.with_fingers:
            self.input_dim_pose = (2 * MOTION_FEATS_BODYHAND)
            fidnet_dim_pose = (2 * BODYHAND_JOINTS * 3)
        else:
            self.input_dim_pose = (2 * MOTION_FEATS_BODY_ONLY)
            fidnet_dim_pose = (2 * BODY_JOINTS * 3)
        self.num_jts = BODYHAND_JOINTS if self.opt.with_fingers else BODY_JOINTS
        self.vq_model, self.vq_opt, self.topcode_transformer_model, self.topcode_trans_opt = self.load_pretrained_models(self.opt.vq_pretrained_weight_path, 
                                                                                                                    self.opt.topcode_transformer_weight_path)
        self.traj_model, self.traj_opt = self.load_pretrained_traj_models(traj_pred_model_pretrained_weight_path) 
        
        self.frames = self.opt.window_size // self.opt.downsample_rate
        self.batch_size = self.opt.batch_size

        self.opt.num_tokens = self.vq_opt.code_num
        self.opt.code_dim = self.vq_opt.code_dim
        self.opt.topcode_dim = 2 * self.vq_opt.code_dim
        print('code num: ', self.opt.num_tokens)
        print('bottom code dim: ', self.opt.code_dim)
        print('top code dim: ', self.opt.topcode_dim)
        # instantiate the transformer model and move it to the right device
        net = Music2Motion_combined_hier_bottom( 
            input_feats=MOTION_FEATS_BODY_ONLY, 
            genre_emb_dim=self.opt.genre_emb_dim, 
            num_tokens=self.opt.num_tokens,
            code_dim=self.opt.code_dim,   
            latent_dim=self.opt.latent_dim,
            ff_size=self.opt.ff_size,
            num_layers=self.opt.n_layers,
            num_heads=self.opt.n_heads,
            dropout=self.opt.dropout,
            music_feats_dim=self.opt.music_feats_dim,
            norm=self.opt.norm,
            activation=self.opt.activation,
            noise_schedule=self.opt.noise_schedule,
            num_genres=self.opt.num_genres, 
            p_emb_dim=self.opt.p_emb_dim,
            input_motion_feats=self.input_dim_pose//2, 
            music_feats_len=self.opt.window_size,
            gpu_id=self.opt.gpu_id[0]
            )

        self.device = torch.device('cuda:' + str(self.opt.gpu_id[0]) if torch.cuda.is_available() and sum(self.opt.gpu_id) > -1 else 'cpu')
        self.opt.device = self.device
        self.transformer_model = net.to(self.device)
        pc_vq = sum(param.numel() for param in net.parameters())
        print('Total parameters of Transformer: {:,}'.format(pc_vq))
        self.epoch = 0
        self.it = 0 
        if self.opt.load_exp is not None:
            print('Loading pre-trained model')
            self.epoch, self.it = self.resume(self.opt.load_exp, change_lr=self.opt.change_lr)
            print('Load model epoch: {}, iterations: {}'.format(self.epoch, self.it))
        

        self.load_dataset = DD100_dataset(data_root=self.opt.dataset_root, split=test_split, window_size=self.opt.window_size, 
                                    with_fingers=True, predict_velocity=self.opt.predict_velocity,
                                    data_scale=self.opt.data_scale,
                                    downsample_rate=self.opt.downsample_rate, device=self.opt.device)
        self.data_loader = DataLoader(self.load_dataset, batch_size=1, pin_memory=False,
                                    drop_last=True, num_workers=self.opt.num_workers,
                                    shuffle=False)
        self.deltamean = to_tensor(np.load(os.path.join(self.opt.dataset_root, 'Deltatrans_rel_Mean.npy'))).to(self.device)
        self.deltastd = to_tensor(np.load(os.path.join(self.opt.dataset_root, 'Deltatrans_rel_Std.npy'))).to(self.device)
        
    
    def inv_z_normalization_delta(self, data):
        return data * self.deltastd + self.deltamean

    def load_pretrained_traj_models(self, trajpred_model_weight_path):
        opt_path = os.path.join(os.path.dirname(os.path.dirname(trajpred_model_weight_path)), 'opt.txt')
        trajpred_opt = get_opt(opt_path)
        trajpred_opt.gpu_id = self.opt.gpu_id
        traj_pred_model = eval(trajpred_opt.model_name)( 
            input_feats=MOTION_FEATS_BODY_ONLY - 3,    
            output_feats= 6,    
            output_emb_width=trajpred_opt.output_emb_width,
            down_t=trajpred_opt.down_t,
            stride_t=trajpred_opt.stride_t,
            width=trajpred_opt.width,
            depth=trajpred_opt.depth,
            dilation_growth_rate=trajpred_opt.dilation_growth_rate,
            norm=trajpred_opt.norm,
            activation=trajpred_opt.act,
            gpu_id=trajpred_opt.gpu_id[0]
            )
        trajpred_opt.device = torch.device('cuda:' + str(self.opt.gpu_id[0]) if torch.cuda.is_available() and sum(trajpred_opt.gpu_id) > -1 else 'cpu')        
        traj_pred_model = traj_pred_model.to(trajpred_opt.device)
        checkpoint = torch.load(trajpred_model_weight_path, map_location=trajpred_opt.device, weights_only=True)
        traj_pred_model.load_state_dict(checkpoint['model'], strict=True)
        traj_pred_model = freeze_model(traj_pred_model)
        print(f'Loaded Trajectory prediction Model {trajpred_opt.model_name}')
        return traj_pred_model, trajpred_opt

    def load_pretrained_models(self, vq_pretrained_weight_path, topcode_transformer_pretrained_weight_path):
        vq_opt_path = os.path.join(os.path.dirname(os.path.dirname(vq_pretrained_weight_path)), 'opt.txt')
        trans_opt_path = os.path.join(os.path.dirname(os.path.dirname(topcode_transformer_pretrained_weight_path)), 'opt.txt')
        vq_opt = get_opt(vq_opt_path)
        trans_opt = get_opt(trans_opt_path)
        vq_opt.gpu_id = self.opt.gpu_id
        trans_opt.gpu_id = self.opt.gpu_id

        vq_model = eval(vq_opt.model_name)( 
            input_feats=self.input_dim_pose,    
            output_feats=self.input_dim_pose,    
            quantizer=vq_opt.quantizer,
            code_num=vq_opt.code_num,
            code_dim=vq_opt.code_dim,
            output_emb_width = vq_opt.output_emb_width,
            down_t=vq_opt.down_t,
            stride_t=vq_opt.stride_t,
            width=vq_opt.width,
            depth=vq_opt.depth,
            dilation_growth_rate=vq_opt.dilation_growth_rate,
            norm=vq_opt.vq_norm,
            activation=vq_opt.vq_act,
            num_genres=vq_opt.num_genres, 
            genre_emb_dim=vq_opt.genre_emb_dim, 
            p_emb_dim=vq_opt.p_emb_dim, 
            gpu_id=vq_opt.gpu_id[0]
            )
        topcode_model = eval(trans_opt.model_name)( 
            input_feats=MOTION_FEATS_BODY_ONLY, 
            genre_emb_dim=trans_opt.genre_emb_dim, 
            num_tokens=vq_opt.code_num,
            code_dim=2*vq_opt.code_dim,   
            latent_dim=trans_opt.latent_dim,
            ff_size=trans_opt.ff_size,
            num_layers=trans_opt.n_layers,
            num_heads=trans_opt.n_heads,
            dropout=trans_opt.dropout,
            music_feats_dim=trans_opt.music_feats_dim,
            norm=trans_opt.norm,
            activation=trans_opt.activation,
            noise_schedule=trans_opt.noise_schedule,
            num_genres=trans_opt.num_genres, 
            p_emb_dim=trans_opt.p_emb_dim,
            input_motion_feats=self.input_dim_pose//2, 
            music_feats_len=trans_opt.window_size,
            gpu_id=trans_opt.gpu_id[0]
            )
        vq_opt.device = torch.device('cuda:' + str(vq_opt.gpu_id[0]) if torch.cuda.is_available() and sum(vq_opt.gpu_id) > -1 else 'cpu')
        trans_opt.device = torch.device('cuda:' + str(trans_opt.gpu_id[0]) if torch.cuda.is_available() and sum(trans_opt.gpu_id) > -1 else 'cpu')
        
        vq_model = vq_model.to(vq_opt.device)
        topcode_model = topcode_model.to(trans_opt.device)

        checkpoint_vq = torch.load(vq_pretrained_weight_path, map_location=vq_opt.device, weights_only=True)
        vq_model.load_state_dict(checkpoint_vq['vq_model'], strict=True)
        vq_model = freeze_model(vq_model)
        print(f'Loaded VQ Model {vq_opt.model_name}')

        checkpoint_tr = torch.load(topcode_transformer_pretrained_weight_path, map_location=trans_opt.device, weights_only=True)
        topcode_model.load_state_dict(checkpoint_tr['vq_model'], strict=True)
        topcode_model = freeze_model(topcode_model)
        print(f'Loaded Topcode Transformer Model {trans_opt.model_name}')
        return vq_model, vq_opt, topcode_model, trans_opt
 

    def save_npz(self): 
        self.transformer_model.eval()
        self.topcode_transformer_model.eval()
        self.vq_model.eval()
        self.traj_model.eval()
        remasking_iterations = 20
        for running_iter, batch_data in enumerate(tqdm(self.data_loader)):
            if running_iter % 2 != 0:       # remove the augmented mirrored motion samples 
                continue
            pred_top_code_idx  = self.topcode_transformer_model.generate(batch_data['music_feats_mfcc'],
                                                                    batch_data['music_feats_chroma'],
                                                                    token_len=25,
                                                                    timesteps=remasking_iterations
                                                                    )
            
            pred_bottom_code_idx  = self.transformer_model.generate(pred_top_code_idx,
                                                            batch_data['music_feats_mfcc'],
                                                            batch_data['music_feats_chroma'],
                                                            token_len=50,
                                                            timesteps=remasking_iterations
                                                            )
            
            pred_code_idx = [pred_bottom_code_idx, pred_top_code_idx]
            pred_motion_delta = self.vq_model.decode(pred_code_idx, batch_data['genre_class'], batch_data['music_feats_mfcc'], batch_data['music_feats_chroma'])
            
            # Try the trajectory predictor here
            input_feats_traj_pred = torch.cat((pred_motion_delta[..., ROOTTRANS_END_INDEX:pred_motion_delta.shape[-1]//2].unsqueeze(-1),
                                        pred_motion_delta[..., ROOTTRANS_END_INDEX+pred_motion_delta.shape[-1]//2:].unsqueeze(-1)), dim=-1).to(self.device)
            pred_root_delta = self.traj_model(input_feats_traj_pred)
            pred_motion_delta = torch.cat((pred_root_delta[..., :ROOTTRANS_END_INDEX],
                                           pred_motion_delta[..., ROOTTRANS_END_INDEX:pred_motion_delta.shape[-1]//2], 
                                           pred_root_delta[..., ROOTTRANS_END_INDEX:],
                                           pred_motion_delta[..., ROOTTRANS_END_INDEX+pred_motion_delta.shape[-1]//2:]), dim=-1).to(self.device)
            
            pred_motion_delta_unnormalized = self.inv_z_normalization_delta(pred_motion_delta)
            pred_p0_delta_unnormalized = pred_motion_delta_unnormalized[..., :pred_motion_delta_unnormalized.shape[-1]//2]
            pred_p1_delta_unnormalized = pred_motion_delta_unnormalized[..., pred_motion_delta_unnormalized.shape[-1]//2:]
            predict_p0_unnormalized, predict_p1_unnormalized = delta_rel_trans_mot2mot_feats(batch_data['p0_init'], 
                                                                                        pred_p0_delta_unnormalized,
                                                                                        pred_p1_delta_unnormalized)

            predict_p0 = mot_to_smplx(predict_p0_unnormalized[0], betas=batch_data['p0_betas'][0], gender='male' if batch_data['p0_gender'][0] == 1 else 'female')
            predict_p1 = mot_to_smplx(predict_p1_unnormalized[0], betas=batch_data['p1_betas'][0], gender='male' if batch_data['p1_gender'][0] == 1 else 'female')
            pkl_savepath = makepath(os.path.join(os.path.dirname(load_exp), 'pkl_it_' + str(remasking_iterations),
                                                 os.path.basename(batch_data['sequence_name'][0])), isfile=True)
            gen_dict = {
                'p0': predict_p0,
                'p1': predict_p1,
            }
            with open(pkl_savepath, 'wb') as handle:
                pickle.dump(gen_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    global opt
    opt = arg_parse(is_train=False, load_exp=load_exp)
    fixseed(opt.seed)
    trainer = MaskedTransformerTrainer(opt)  
    trainer.save_npz()