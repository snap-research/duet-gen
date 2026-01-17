import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
sys.path.append('.')
sys.path.append('..')
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
import tqdm
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from body_models.smplx.body_models import SMPLXLayer as smplxlayer
from dataset_preparation.dd100_loader import DD100_dataset_3 as DD100_dataset
from dataset_preparation.data_utils import *
from models.vqvae.motion_vqvae import *
from train.args_motion_vqvae import arg_parse
from tools.joint_names import *
from utils.transformations import *
from utils.utils import *

is_train=True
is_evaluate = False
test_split = 'test'  # the dataset split that you want to evaluate on.
load_exp = None
do_visualize = True

def dataset_prepare(opt):
    train_dataset = DD100_dataset(data_root=opt.dataset_root, split='train', window_size=opt.window_size,  
                                  with_fingers=opt.with_fingers, predict_velocity=opt.predict_velocity,
                                  data_scale=opt.data_scale,
                                  downsample_rate=opt.downsample_rate, device=opt.device)
    val_dataset = DD100_dataset(data_root=opt.dataset_root, split='test', window_size=opt.window_size, 
                                with_fingers=opt.with_fingers, predict_velocity=opt.predict_velocity,
                                data_scale=opt.data_scale,
                                downsample_rate=opt.downsample_rate, device=opt.device)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, pin_memory=False, 
                              drop_last=True, num_workers=opt.num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, pin_memory=False,
                            drop_last=True, num_workers=opt.num_workers,
                            shuffle=False)
    return train_loader, val_loader

def def_value():
    return 0.0
def large_value():
    return 1e+5

class VQTokenizerTrainer:
    def __init__(self, args):
        self.opt = args
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print('Using ', self.opt.gpu_id, ' GPUs!')
        self.num_jts = BODYHAND_JOINTS if self.opt.with_fingers else BODY_JOINTS
        self.frames = self.opt.window_size // self.opt.downsample_rate
        self.batch_size = self.opt.batch_size
        if self.opt.with_fingers:
            dim_pose = (2 * MOTION_FEATS_BODYHAND)
        else:
            dim_pose = (2 * MOTION_FEATS_BODY_ONLY)
    
        # instantiate the model and move it to the right device
        net = Hierarchical_VQVAE_TwoPerson( 
            # args=self.opt,
            input_feats=dim_pose,    
            output_feats=dim_pose,    
            quantizer=self.opt.quantizer,
            code_num=self.opt.code_num,
            code_dim=self.opt.code_dim,
            output_emb_width = self.opt.output_emb_width,
            down_t=self.opt.down_t,
            stride_t=self.opt.stride_t,
            width=self.opt.width,
            depth=self.opt.depth,
            dilation_growth_rate=self.opt.dilation_growth_rate,
            norm=self.opt.vq_norm,
            activation=self.opt.vq_act,
            num_genres=self.opt.num_genres, 
            genre_emb_dim=self.opt.genre_emb_dim, 
            p_emb_dim=self.opt.p_emb_dim, 
            gpu_id=self.opt.gpu_id[0]
            )
        self.device = torch.device('cuda:' + str(self.opt.gpu_id[0]) if torch.cuda.is_available() and sum(self.opt.gpu_id) > -1 else 'cpu')
        print("using device:", self.device)
        self.opt.device = self.device
        # net = torch.nn.DataParallel(net, device_ids=self.opt.gpu_id)
        self.vq_model = net.to(self.device)
        pc_vq = sum(param.numel() for param in net.parameters())
        print('Total parameters of VQVAE: {:,}'.format(pc_vq))
        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt_vq_model, step_size=self.opt.step_size, gamma=self.opt.gamma)
        self.epoch = 0
        self.it = 0 
        if self.opt.load_exp is not None:
            print('Loading pre-trained model')
            self.epoch, self.it = self.resume(self.opt.load_exp, change_lr=self.opt.change_lr)
            print('Load model epoch: {}, iterations: {}'.format(self.epoch, self.it))
        
        self.male_smplxmodel = smplxlayer(model_path='models/smplx',
                                gender='male', 
                                num_betas=10,
                                use_pca=False,
                                use_face_contour=True, 
                                flat_hand_mean=True).to(self.device)

        
        self.female_smplxmodel = smplxlayer(model_path='models/smplx',
                                gender='female', 
                                num_betas=10,
                                use_pca=False,
                                use_face_contour=True, 
                                flat_hand_mean=True).to(self.device)
        
        
        if self.opt.is_train:
            self.train_loader, self.val_loader = dataset_prepare(self.opt)
            # self.inv_z_normalization = self.train_loader.dataset.inv_z_normalization
            self.inv_z_normalization_delta = self.train_loader.dataset.inv_z_normalization_delta
            self.logger = SummaryWriter(log_dir=os.path.join(self.opt.logs_dir, str(datetime.now()).replace(" ", "_")))
            l1_criterion = None
            if self.opt.recons_loss == 'l1_smooth':
                l1_criterion = torch.nn.SmoothL1Loss()
            elif self.opt.recons_loss == 'l1':
                l1_criterion = torch.nn.L1Loss()

            self.l1_smoothloss = torch.nn.SmoothL1Loss()
            self.l1_loss = torch.nn.L1Loss()
            self.mse_loss = torch.nn.MSELoss()
            self.bce_ = torch.nn.BCELoss()
            self.sigmoid_ = torch.nn.Sigmoid()
            self.rec_loss_list = [l1_criterion, l1_criterion, l1_criterion, l1_criterion, 
                                  l1_criterion, l1_criterion, l1_criterion, l1_criterion, l1_criterion, l1_criterion, l1_criterion]
            self.rel_rec_loss_list = [l1_criterion, l1_criterion, l1_criterion, l1_criterion, l1_criterion, l1_criterion, l1_criterion, l1_criterion]
            self.vel_loss_list = [l1_criterion, l1_criterion, l1_criterion, l1_criterion, 
                                  l1_criterion, l1_criterion, l1_criterion, l1_criterion, l1_criterion, l1_criterion, l1_criterion]
        else:
            self.load_dataset = DD100_dataset(data_root=self.opt.dataset_root, split=test_split, window_size=self.opt.window_size, 
                                     with_fingers=self.opt.with_fingers, predict_velocity=self.opt.predict_velocity,
                                     data_scale=self.opt.data_scale,
                                     downsample_rate=self.opt.downsample_rate, device=self.opt.device)
            self.data_loader = DataLoader(self.load_dataset, batch_size=1, pin_memory=False,
                                        drop_last=False, num_workers=self.opt.num_workers,
                                        shuffle=False)
            # self.inv_z_normalization = self.data_loader.dataset.inv_z_normalization
            self.inv_z_normalization_delta = self.data_loader.dataset.inv_z_normalization_delta

    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, dir_name, ep, total_it):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        # Also save the train test loss
        torch.save(state, makepath(os.path.join(dir_name, 'weights.p'), isfile=True))
  

    def resume(self, model_path, change_lr=False):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'], strict=True)
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        if change_lr:
            for param_group in self.opt_vq_model.param_groups:
                param_group["lr"] = self.opt.lr
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    
    def interaction_dist_contacts_wrists(self, p0_global_pos_wrists, p1_global_pos_wrists, predicted_p0_global_pos_wrists, predicted_p1_global_pos_wrists):
        B, T, J, D = p0_global_pos_wrists.shape
        p0_global_pos = p0_global_pos_wrists.repeat(1, 1, J, 1)
        p1_global_pos = torch.repeat_interleave(p1_global_pos_wrists, J, 2)
        predicted_p0_global_pos = predicted_p0_global_pos_wrists.repeat(1, 1, J, 1)
        predicted_p1_global_pos = torch.repeat_interleave(predicted_p1_global_pos_wrists, J, 2)
        rel_global_pos_weight = 2*torch.exp(-1*(p1_global_pos - p0_global_pos).norm(dim=-1, keepdim=True))
        rel_global_pos = rel_global_pos_weight*(p1_global_pos - p0_global_pos)
        pred_rel_global_pos = rel_global_pos_weight*(predicted_p1_global_pos - predicted_p0_global_pos)
        return rel_global_pos, pred_rel_global_pos

    def interaction_dist_contacts(self, p0_global_pos_, p1_global_pos_, predicted_p0_global_pos_, predicted_p1_global_pos_):
        B, T, J, D = p0_global_pos_.shape
        p0_global_pos = torch.repeat_interleave(p0_global_pos_, J, 2)
        p1_global_pos = p1_global_pos_.repeat(1, 1, J, 1)
        predicted_p0_global_pos = torch.repeat_interleave(predicted_p0_global_pos_, J, 2)
        predicted_p1_global_pos = predicted_p1_global_pos_.repeat(1, 1, J, 1)
        rel_global_pos_weight = 2*torch.exp(-1*(p1_global_pos - p0_global_pos).norm(dim=-1))
        rel_global_pos = rel_global_pos_weight*(p1_global_pos - p0_global_pos).norm(dim=-1)
        random = torch.randint(0, 3, (1, 1))
        if random == 0:
            pred_rel_global_pos = rel_global_pos_weight*(predicted_p1_global_pos - predicted_p0_global_pos).norm(dim=-1)
        elif random == 1:
            pred_rel_global_pos = rel_global_pos_weight*(predicted_p1_global_pos - p0_global_pos).norm(dim=-1)
        else:
            pred_rel_global_pos = rel_global_pos_weight*(p1_global_pos - predicted_p0_global_pos).norm(dim=-1)
        return rel_global_pos, pred_rel_global_pos

    def forward(self, batch_data, iterations, teacher_forcing):
        loss_model = 0.0
        delta_rec_loss = torch.Tensor([0.0])
        if self.opt.predict_velocity:
            input_delta_motion = torch.cat((batch_data['p0_motion'], batch_data['p1_motion']), dim=-1).to(self.device)
            pred_motion_delta, loss_commit, perplexity, code_idx = self.vq_model(input_delta_motion, batch_data['genre_class'], batch_data['music_feats_mfcc'], batch_data['music_feats_chroma'])
            delta_rec_loss = self.opt.weight_loss_rec[-1] * self.l1_smoothloss(pred_motion_delta, input_delta_motion)
            loss_model += delta_rec_loss
            pred_motion_delta_unnormalized = self.inv_z_normalization_delta(pred_motion_delta)
            pred_p0_delta_unnormalized = pred_motion_delta_unnormalized[..., :pred_motion_delta_unnormalized.shape[-1]//2]
            pred_p1_delta_unnormalized = pred_motion_delta_unnormalized[..., pred_motion_delta_unnormalized.shape[-1]//2:]
           
            predict_p0_unnormalized, predict_p1_unnormalized = delta_rel_trans_mot2mot_feats(batch_data['p0_init'], 
                                                                                             pred_p0_delta_unnormalized,
                                                                                             pred_p1_delta_unnormalized)

            pred_motion = torch.cat((predict_p0_unnormalized, predict_p1_unnormalized), dim=-1)
            p0_root_transl, p0_root_rot, p0_body_rot, p0_local_jointpos, p0_local_jointvel, p0_foot_contact = reverse_process_mot(batch_data['p0_motion_unnormalized'], with_fingers=self.opt.with_fingers)
            p1_root_transl, p1_root_rot, p1_body_rot, p1_local_jointpos, p1_local_jointvel, p1_foot_contact = reverse_process_mot(batch_data['p1_motion_unnormalized'], with_fingers=self.opt.with_fingers)
            rel_features = [batch_data['p1_motion_unnormalized'][..., :ROOTTRANS_END_INDEX] - batch_data['p0_motion_unnormalized'][..., :ROOTTRANS_END_INDEX],
                        batch_data['p1_motion_unnormalized'][..., ROOTTRANS_END_INDEX:ROOTROT_END_INDEX] - batch_data['p0_motion_unnormalized'][..., ROOTTRANS_END_INDEX:ROOTROT_END_INDEX]
                        ]
        
        loss_commit = torch.mean(loss_commit)
        perplexity = torch.mean(perplexity)
        
        B, T, dim = pred_motion.shape
        
        p0_features = [p0_root_transl, p0_root_rot, p0_body_rot, p0_local_jointpos, p0_local_jointvel, p0_foot_contact]
        p1_features = [p1_root_transl, p1_root_rot, p1_body_rot, p1_local_jointpos, p1_local_jointvel, p1_foot_contact]
        if self.opt.with_fingers:
            predict_p0 = pred_motion[..., :MOTION_FEATS_BODYHAND]
            predict_p1 = pred_motion[..., MOTION_FEATS_BODYHAND:2*MOTION_FEATS_BODYHAND]
        else:
            predict_p0 = pred_motion[..., :MOTION_FEATS_BODY_ONLY]
            predict_p1 = pred_motion[..., MOTION_FEATS_BODY_ONLY:2*MOTION_FEATS_BODY_ONLY]
        predict_rel_pos = predict_p1[..., :ROOTTRANS_END_INDEX] - predict_p0[..., :ROOTTRANS_END_INDEX]
        predict_rel_rot = predict_p1[..., ROOTTRANS_END_INDEX:ROOTROT_END_INDEX] - predict_p0[..., ROOTTRANS_END_INDEX:ROOTROT_END_INDEX]
            
        predict_p0_root_transl, predict_p0_root_rot, predict_p0_body_rot, predict_p0_local_jointpos, predict_p0_local_jointvel, predict_p0_foot_contact = reverse_process_mot(predict_p0, with_fingers=self.opt.with_fingers)
        predict_p1_root_transl, predict_p1_root_rot, predict_p1_body_rot, predict_p1_local_jointpos, predict_p1_local_jointvel, predict_p1_foot_contact = reverse_process_mot(predict_p1, with_fingers=self.opt.with_fingers)
        predict_p0_features = [predict_p0_root_transl, predict_p0_root_rot, predict_p0_body_rot, predict_p0_local_jointpos, predict_p0_local_jointvel, predict_p0_foot_contact]
        predict_p1_features = [predict_p1_root_transl, predict_p1_root_rot, predict_p1_body_rot, predict_p1_local_jointpos, predict_p1_local_jointvel, predict_p1_foot_contact]
        pred_rel_features = [predict_rel_pos, predict_rel_rot]
        
            
        if iterations > 70000: 
            
            predict_smplx_params_p0 = batch_mot_to_smplx(predict_p0_unnormalized, batch_data['p0_betas'], with_fingers=self.opt.with_fingers,
                                                 device=self.device)
            predict_smplx_params_p1 = batch_mot_to_smplx(predict_p1_unnormalized, batch_data['p1_betas'], with_fingers=self.opt.with_fingers,
                                                 device=self.device)
            
            p0_males = batch_data['p0_gender'] == 1
            p0_females = ~p0_males
            p1_males = batch_data['p1_gender'] == 1
            p1_females = ~p1_males
            predicted_p0_global_pos = torch.zeros((self.batch_size, self.frames, self.num_jts, 3)).to(self.device)
            predicted_p1_global_pos = torch.zeros((self.batch_size, self.frames, self.num_jts, 3)).to(self.device)
            if sum(p0_females) > 0:
                pred_f_smplx_params_p0 = {k: v[p0_females] for k, v in predict_smplx_params_p0.items()}
                pred_f_smplx_params_p0 = smplx_batchify(pred_f_smplx_params_p0)
                predicted_p0_global_pos[p0_females] = (self.female_smplxmodel(
                    **pred_f_smplx_params_p0))[:, :self.num_jts].reshape(-1, self.frames, self.num_jts, 3)
            if sum(p0_males) > 0:
                pred_m_smplx_params_p0 = {k: v[p0_males] for k, v in predict_smplx_params_p0.items()}
                pred_m_smplx_params_p0 = smplx_batchify(pred_m_smplx_params_p0)
                predicted_p0_global_pos[p0_males] = (self.male_smplxmodel(
                    **pred_m_smplx_params_p0))[:, :self.num_jts].reshape(-1, self.frames, self.num_jts, 3)
            if sum(p1_females) > 0:
                pred_f_smplx_params_p1 = {k: v[p1_females] for k, v in predict_smplx_params_p1.items()}
                pred_f_smplx_params_p1 = smplx_batchify(pred_f_smplx_params_p1)
                predicted_p1_global_pos[p1_females] = (self.female_smplxmodel(
                    **pred_f_smplx_params_p1))[:, :self.num_jts].reshape(-1, self.frames, self.num_jts, 3)
            if sum(p1_males) > 0:
                pred_m_smplx_params_p1 = {k: v[p1_males] for k, v in predict_smplx_params_p1.items()}
                pred_m_smplx_params_p1 = smplx_batchify(pred_m_smplx_params_p1)
                predicted_p1_global_pos[p1_males] = (self.male_smplxmodel(
                    **pred_m_smplx_params_p1))[:, :self.num_jts].reshape(-1, self.frames, self.num_jts, 3)
            p0_global_pos = (batch_data['p0_globalpos'].reshape(-1, self.frames, BODYHAND_JOINTS, 3))[:, :, :self.num_jts]
            p1_global_pos = (batch_data['p1_globalpos'].reshape(-1, self.frames, BODYHAND_JOINTS, 3))[:, :, :self.num_jts]
            p0_features.append(p0_global_pos)
            p1_features.append(p1_global_pos)
            predict_p0_features.append(predicted_p0_global_pos)
            predict_p1_features.append(predicted_p1_global_pos)
            
            
            # use a weight decay interaction loss (decide whether to use 1-1 or 1-many)
            rel_global_pos, pred_rel_global_pos = self.interaction_dist_contacts(p0_global_pos, p1_global_pos, predicted_p0_global_pos, predicted_p1_global_pos)
            rel_features.append(rel_global_pos)
            pred_rel_features.append(pred_rel_global_pos)
            
            # special attention to wrist joints
            p0_features.append(p0_global_pos[:, :, wrist_jts])
            p1_features.append(p1_global_pos[:, :, wrist_jts])
            predict_p0_features.append(predicted_p0_global_pos[:, :, wrist_jts])
            predict_p1_features.append(predicted_p1_global_pos[:, :, wrist_jts])
            rel_global_pos_wrists, pred_rel_global_pos_wrists = self.interaction_dist_contacts_wrists(p0_global_pos[:, :, wrist_jts], 
                                                                                        p1_global_pos[:, :, wrist_jts],
                                                                                        predicted_p0_global_pos[:, :, wrist_jts], 
                                                                                        predicted_p1_global_pos[:, :, wrist_jts])
            rel_features.append(rel_global_pos_wrists)
            pred_rel_features.append(pred_rel_global_pos_wrists)
            
            # special attention to feet joints
            p0_features.append(p0_global_pos[:, :, feet_jts])
            p1_features.append(p1_global_pos[:, :, feet_jts])
            predict_p0_features.append(predicted_p0_global_pos[:, :, feet_jts])
            predict_p1_features.append(predicted_p1_global_pos[:, :, feet_jts])
             
        num_features = len(p0_features)
        loss_rec_p0 = [self.opt.weight_loss_rec[i] * self.rec_loss_list[i](predict_p0_features[i], p0_features[i])
                       for i in range(num_features)]
        loss_rec_p1 = [self.opt.weight_loss_rec[i] * self.rec_loss_list[i](predict_p1_features[i], p1_features[i])
                       for i in range(num_features)]
        num_rel_features = len(rel_features)
        loss_rel_rec = [self.opt.weight_loss_relative_pos[i] * self.rel_rec_loss_list[i](pred_rel_features[i], rel_features[i])
                        for i in range(num_rel_features)]
        
        if iterations < 20000:
            loss_vel_p0 = [self.opt.weight_loss_vel[i] * self.vel_loss_list[i](predict_p0_features[i][:, 20:] - predict_p0_features[i][:, :-20], p0_features[i][:, 20:] - p0_features[i][:, :-20])
                        for i in range(num_features)]
            loss_vel_p1 = [self.opt.weight_loss_vel[i] * self.vel_loss_list[i](predict_p1_features[i][:, 20:] - predict_p1_features[i][:, :-20], p1_features[i][:, 20:] - p1_features[i][:, :-20])
                        for i in range(num_features)]
            loss_rel_vel = [self.opt.weight_loss_relative_vel[i] * self.vel_loss_list[i](pred_rel_features[i][:, 20:] - pred_rel_features[i][:, :-20],
                                                    rel_features[i][:, 20:] - rel_features[i][:, :-20]) for i in range(num_rel_features)]
        else:
            loss_vel_p0 = [self.opt.weight_loss_vel[i] * self.vel_loss_list[i](predict_p0_features[i][:, 5:] - predict_p0_features[i][:, :-5], p0_features[i][:, 5:] - p0_features[i][:, :-5])
                        for i in range(num_features)]
            loss_vel_p1 = [self.opt.weight_loss_vel[i] * self.vel_loss_list[i](predict_p1_features[i][:, 5:] - predict_p1_features[i][:, :-5], p1_features[i][:, 5:] - p1_features[i][:, :-5])
                        for i in range(num_features)]
            loss_rel_vel = [self.opt.weight_loss_relative_vel[i] * self.vel_loss_list[i](pred_rel_features[i][:, 5:] - pred_rel_features[i][:, :-5],
                                                    rel_features[i][:, 5:] - rel_features[i][:, :-5]) for i in range(num_rel_features)]
        
        
        
        for i_loss in loss_rec_p0:
            loss_model += i_loss
        for i_loss in loss_rec_p1:
            loss_model += i_loss
        for i_loss in loss_vel_p0:
            loss_model += i_loss    
        for i_loss in loss_vel_p1:
            loss_model += i_loss
        for i_loss in loss_rel_rec:
            loss_model += i_loss    
        for i_loss in loss_rel_vel:
            loss_model += i_loss

        loss_model = loss_model + self.opt.weight_loss_commit * loss_commit

        return pred_motion, loss_model, [loss_rec_p0, loss_rec_p1, loss_vel_p0, loss_vel_p1, loss_rel_rec, loss_rel_vel, delta_rec_loss], loss_commit, perplexity

    def loss_logs_arrange(self, loss_logs, loss_list):
        loss_logs['loss_rec_p0_root_transl'] += loss_list[0][0].item()
        loss_logs['loss_rec_p0_root_rot'] += loss_list[0][1].item()
        loss_logs['loss_rec_p0_body_rot'] += loss_list[0][2].item()
        loss_logs['loss_rec_p0_local_jointpos'] += loss_list[0][3].item()
        loss_logs['loss_rec_p0_local_jointvel'] += loss_list[0][4].item()
        loss_logs['loss_rec_p0_foot_contact'] += loss_list[0][5].item()
        loss_logs['loss_rec_p1_root_transl'] += loss_list[1][0].item()
        loss_logs['loss_rec_p1_root_rot'] += loss_list[1][1].item()
        loss_logs['loss_rec_p1_body_rot'] += loss_list[1][2].item()
        loss_logs['loss_rec_p1_local_jointpos'] += loss_list[1][3].item()
        loss_logs['loss_rec_p1_local_jointvel'] += loss_list[1][4].item()
        loss_logs['loss_rec_p1_foot_contact'] += loss_list[1][5].item()
        loss_logs['loss_vel_p0_root_transl'] += loss_list[2][0].item()
        loss_logs['loss_vel_p0_root_rot'] += loss_list[2][1].item()
        loss_logs['loss_vel_p0_body_rot'] += loss_list[2][2].item()
        loss_logs['loss_vel_p0_local_jointpos'] += loss_list[2][3].item()
        loss_logs['loss_vel_p0_local_jointvel'] += loss_list[2][4].item()
        loss_logs['loss_vel_p0_foot_contact'] += loss_list[2][5].item()
        loss_logs['loss_vel_p1_root_transl'] += loss_list[3][0].item()
        loss_logs['loss_vel_p1_root_rot'] += loss_list[3][1].item()
        loss_logs['loss_vel_p1_body_rot'] += loss_list[3][2].item()
        loss_logs['loss_vel_p1_local_jointpos'] += loss_list[3][3].item()
        loss_logs['loss_vel_p1_local_jointvel'] += loss_list[3][4].item()
        loss_logs['loss_vel_p1_foot_contact'] += loss_list[3][5].item()
        loss_logs['loss_rel_trans'] += loss_list[4][0].item()
        loss_logs['loss_rel_rot'] += loss_list[4][1].item()
        loss_logs['loss_vel_rel_trans'] += loss_list[5][0].item()
        loss_logs['loss_vel_rel_rot'] += loss_list[5][1].item()
        loss_logs['delta_rec_loss'] += loss_list[6].item()
        if len(loss_list[0]) > 6:
            loss_logs['loss_rec_p0_global_pos'] += loss_list[0][6].item()
            loss_logs['loss_rec_p1_global_pos'] += loss_list[1][6].item()
            loss_logs['loss_vel_p0_global_pos'] += loss_list[2][6].item()
            loss_logs['loss_vel_p1_global_pos'] += loss_list[3][6].item()
            loss_logs['loss_rel_pos'] += loss_list[4][2].item()
            loss_logs['loss_vel_rel_pos'] += loss_list[5][2].item()
        if len(loss_list[0]) > 7:
            loss_logs['loss_rec_p0_global_wrist'] += loss_list[0][7].item()
            loss_logs['loss_rec_p1_global_wrist'] += loss_list[1][7].item()
            loss_logs['loss_vel_p0_global_wrist'] += loss_list[2][7].item()
            loss_logs['loss_vel_p1_global_wrist'] += loss_list[3][7].item()
            loss_logs['loss_rel_wrist'] += loss_list[4][3].item()
            loss_logs['loss_vel_rel_wrist'] += loss_list[5][3].item()
        return loss_logs
    
    def train_epoch(self, train_loader, epoch, it, teacher_forcing):
        self.vq_model.train()
        train_logs = defaultdict(def_value, OrderedDict())
        mean_loss = defaultdict(def_value, OrderedDict())

        for running_iter, batch_data in enumerate(tqdm(train_loader)):
            self.opt_vq_model.zero_grad()      
            
            pred_motion, loss_model, loss_list, loss_commit, perplexity = self.forward(batch_data, it, teacher_forcing)
            if loss_model == float('inf') or torch.isnan(loss_model):
                print('Train loss is nan')
                exit()
           
            train_logs['loss'] += loss_model.item()
            train_logs = self.loss_logs_arrange(train_logs, loss_list)
            train_logs['loss_commit'] += loss_commit.item()
            train_logs['perplexity'] += perplexity.item()
            train_logs['lr'] += self.opt_vq_model.param_groups[0]['lr']
            
            loss_model.mean().backward()
            torch.nn.utils.clip_grad_value_(self.vq_model.parameters(), 0.1)
            self.opt_vq_model.step()
            it += 1
            
        for tag, value in train_logs.items():
            mean_loss[tag] = value / (running_iter + 1)   
        return mean_loss, it

    
    def val_epoch(self, val_loader, epoch, it, teacher_forcing):
        self.vq_model.eval()
        val_logs = defaultdict(def_value, OrderedDict())
        mean_loss = defaultdict(def_value, OrderedDict())
        for running_iter, batch_data in enumerate(tqdm(val_loader)):
            pred_motion, loss_model, loss_list, loss_commit, perplexity = self.forward(batch_data, it, teacher_forcing)
            
            val_logs['loss'] += loss_model.item()
            val_logs = self.loss_logs_arrange(val_logs, loss_list)
            val_logs['loss_commit'] += loss_commit.item()
            val_logs['perplexity'] += perplexity.item()
            val_logs['epoch'] += self.opt_vq_model.param_groups[0]['lr']
            

        for tag, value in val_logs.items():
            mean_loss[tag] = value / (running_iter + 1)   
        return mean_loss

    def train(self):
        self.vq_model.to(self.device)
        total_iters = self.opt.num_epoch * len(self.train_loader)
        print('Total Epochs: {}, Total Iters: {}'.format(self.opt.num_epoch, total_iters))
        current_lr = self.opt.lr
        train_logs = defaultdict(def_value, OrderedDict())
        val_logs = defaultdict(def_value, OrderedDict())
        best_val_loss = 1e+5
        running_epoch = 0
        teacher_forcing = False
        while self.epoch <= self.opt.num_epoch:
            start_time = time.time()
            train_logs, self.it = self.train_epoch(self.train_loader, self.epoch, self.it, teacher_forcing)
            print_current_loss('Train ', os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, self.opt.experiment_name+'_logs.txt'),
                               time.time() - start_time, self.it, train_logs['loss'],
                                epoch=self.epoch,  lr=self.opt_vq_model.param_groups[0]['lr'])
            if running_epoch % self.opt.eval_every_e == 0:
                start_time_ = time.time()
                val_logs = self.val_epoch(self.val_loader, self.epoch, self.it, teacher_forcing)
                print_current_loss('Val ', os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, self.opt.experiment_name+'_logs.txt'),
                                   time.time() - start_time_, self.it, val_logs['loss'], 
                                   epoch=self.epoch, lr=self.opt_vq_model.param_groups[0]['lr'])
                if val_logs['loss'] < best_val_loss:
                    self.save(os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, 'best_val_' ), self.epoch, self.it)
                    best_val_loss = val_logs['loss']
            if running_epoch % self.opt.log_every_e == 0:    
                for tag, value in train_logs.items():
                    self.logger.add_scalar('Train/%s'%tag, train_logs[tag], self.it)
                for tag, value in val_logs.items():
                    self.logger.add_scalar('Val/%s'%tag, val_logs[tag], self.it)
            if running_epoch % self.opt.save_every_e == 0:
                self.save(os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, str(self.epoch)), self.epoch, self.it)
                
            self.scheduler.step() 
            running_epoch += 1
            self.epoch += 1
            self.logger.flush()
        
        self.logger.close()  
        self.save(os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, str(self.epoch)), self.epoch, self.it)
        print('Training complete!')




    def evaluate(self): 
        print("Starting summing and evaluation now")
        self.vq_model.eval()
        p0_mroot_pe_sum = torch.zeros((self.frames)).to(self.device)
        p1_mroot_pe_sum = torch.zeros((self.frames)).to(self.device)
        p0_mroot_re_sum = torch.zeros((self.frames)).to(self.device)
        p1_mroot_re_sum = torch.zeros((self.frames)).to(self.device)
        p0_mpjgpe_sum = torch.zeros((self.frames, self.num_jts)).to(self.device)
        p1_mpjgpe_sum = torch.zeros((self.frames, self.num_jts)).to(self.device)
        p0_mpjrpe_sum = torch.zeros((self.frames, self.num_jts-1)).to(self.device)
        p1_mpjrpe_sum = torch.zeros((self.frames, self.num_jts-1)).to(self.device)
        p0_mpjve_sum = torch.zeros((self.frames-1, self.num_jts)).to(self.device)
        p1_mpjve_sum = torch.zeros((self.frames-1, self.num_jts)).to(self.device) 
        
        for running_iter, batch_data in enumerate(tqdm(self.data_loader)):
            
            if self.opt.predict_velocity:
                input_delta_motion = torch.cat((batch_data['p0_motion'], batch_data['p1_motion']), dim=-1).to(self.device)
                code_idx = self.vq_model.encode(input_delta_motion)
                pred_motion_delta = self.vq_model.decode(code_idx,  batch_data['genre_class'], batch_data['music_feats_mfcc'], batch_data['music_feats_chroma'])

                # inv_z_normalize the predicted deltas and add it with the inits
                pred_motion_delta_unnormalized = self.inv_z_normalization_delta(pred_motion_delta)
                pred_p0_delta_unnormalized = pred_motion_delta_unnormalized[..., :pred_motion_delta_unnormalized.shape[-1]//2]
                pred_p1_delta_unnormalized = pred_motion_delta_unnormalized[..., pred_motion_delta_unnormalized.shape[-1]//2:]
                predict_p0_unnormalized, predict_p1_unnormalized = delta_rel_trans_mot2mot_feats(batch_data['p0_init'], 
                                                                                             pred_p0_delta_unnormalized,
                                                                                             pred_p1_delta_unnormalized)

                timestep = predict_p1_unnormalized.shape[1]
                gt_p0_smplx = mot_to_smplx(batch_data['p0_motion_unnormalized'][0], betas=batch_data['p0_betas'][0], gender='male' if batch_data['p0_gender'][0] == 1 else 'female',
                            with_fingers=self.opt.with_fingers)
                gt_p1_smplx = mot_to_smplx(batch_data['p1_motion_unnormalized'][0], betas=batch_data['p1_betas'][0], gender='male' if batch_data['p1_gender'][0] == 1 else 'female',
                            with_fingers=self.opt.with_fingers)
                pred_p0_smplx = mot_to_smplx(predict_p0_unnormalized[0], betas=batch_data['p0_betas'][0], gender='male' if batch_data['p0_gender'][0] == 1 else 'female',
                            with_fingers=self.opt.with_fingers)
                pred_p1_smplx = mot_to_smplx(predict_p1_unnormalized[0], betas=batch_data['p1_betas'][0], gender='male' if batch_data['p1_gender'][0] == 1 else 'female',
                            with_fingers=self.opt.with_fingers)
                gt_p0_jt = smplx_to_pos3d(gt_p0_smplx)[:, :self.num_jts].to(self.device)
                gt_p1_jt = smplx_to_pos3d(gt_p1_smplx)[:, :self.num_jts].to(self.device)
                gt_p0_rootjt = gt_p0_jt[:, 0]
                gt_p0_root_relative_jt = gt_p0_jt[:, 1:] - gt_p0_jt[:, 0:1]
                gt_p1_rootjt = gt_p1_jt[:, 0]
                gt_p1_root_relative_jt = gt_p1_jt[:, 1:] - gt_p1_jt[:, 0:1]
                pred_p0_jt = smplx_to_pos3d(pred_p0_smplx)[:, :self.num_jts].to(self.device)
                pred_p1_jt = smplx_to_pos3d(pred_p1_smplx)[:, :self.num_jts].to(self.device)
                pred_p0_rootjt = pred_p0_jt[:, 0]
                pred_p0_root_relative_jt = pred_p0_jt[:, 1:] - pred_p0_jt[:, 0:1]
                pred_p1_rootjt = pred_p1_jt[:, 0]
                pred_p1_root_relative_jt = pred_p1_jt[:, 1:] - pred_p1_jt[:, 0:1]
                gt_p0_root_or = to_tensor(gt_p0_smplx['poses'].reshape(timestep, BODYHAND_JOINTS, 3)[:, 0], device=self.device)
                gt_p1_root_or = to_tensor(gt_p1_smplx['poses'].reshape(timestep, BODYHAND_JOINTS, 3)[:, 0], device=self.device)
                pred_p0_root_or = to_tensor(pred_p0_smplx['poses'].reshape(timestep, BODYHAND_JOINTS, 3)[:, 0], device=self.device)
                pred_p1_root_or = to_tensor(pred_p1_smplx['poses'].reshape(timestep, BODYHAND_JOINTS, 3)[:, 0], device=self.device)

                p0_mpjgpe_sum += torch.linalg.norm(pred_p0_jt - gt_p0_jt, dim=-1)
                p1_mpjgpe_sum += torch.linalg.norm(pred_p1_jt - gt_p1_jt, dim=-1)
                p0_mpjrpe_sum += torch.linalg.norm(pred_p0_root_relative_jt - gt_p0_root_relative_jt, dim=-1)
                p1_mpjrpe_sum += torch.linalg.norm(pred_p1_root_relative_jt - gt_p1_root_relative_jt, dim=-1)
                p0_mroot_pe_sum += torch.linalg.norm(pred_p0_rootjt - gt_p0_rootjt, dim=-1)
                p1_mroot_pe_sum += torch.linalg.norm(pred_p1_rootjt - gt_p1_rootjt, dim=-1)
                p0_mroot_re_sum += torch.linalg.norm(pred_p0_root_or - gt_p0_root_or, dim=-1)
                p1_mroot_re_sum += torch.linalg.norm(pred_p1_root_or - gt_p1_root_or, dim=-1)
                p0_mpjve_sum += torch.linalg.norm((pred_p0_jt[1:] - pred_p0_jt[:-1]) - (gt_p0_jt[1:] - gt_p0_jt[:-1]), dim=-1)
                p1_mpjve_sum += torch.linalg.norm((pred_p1_jt[1:] - pred_p1_jt[:-1]) - (gt_p1_jt[1:] - gt_p1_jt[:-1]), dim=-1)
                
                

            if running_iter > 0 and running_iter%50 == 0:
                
                mpjpe_dict = {
                    "p0_mpjgpe" : p0_mpjgpe_sum,
                    "p1_mpjgpe" : p1_mpjgpe_sum,
                    "p0_mpjrpe" : p0_mpjrpe_sum,
                    "p1_mpjrpe" : p1_mpjrpe_sum,
                    "p0_mroot_pe" : p0_mroot_pe_sum,
                    "p1_mroot_pe" : p1_mroot_pe_sum,
                    "p0_mpjve" : p0_mpjve_sum,
                    "p1_mpjve" : p1_mpjve_sum,
                    "p0_mroot_re" : p0_mroot_re_sum,
                    "p1_mroot_re" : p1_mroot_re_sum,
                }
               
                eval_savepath = makepath(os.path.join(os.path.dirname(self.opt.load_exp), 'MPJPE_', str(running_iter) +'.pkl'), isfile=True)
                with open(eval_savepath, 'wb') as f:
                    pickle.dump(mpjpe_dict, f)


if __name__ == "__main__":
    global opt
    opt = arg_parse(is_train=is_train, load_exp=load_exp)
    fixseed(opt.seed)
    
    trainer = VQTokenizerTrainer(opt)
    if is_train:
        trainer.train()
    else: 
        if is_evaluate:
            trainer.evaluate()
        