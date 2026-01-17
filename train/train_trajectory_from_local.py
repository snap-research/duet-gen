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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset_preparation.dd100_loader import DD100_dataset_3 as DD100_dataset
from dataset_preparation.data_utils import *
from models.extra_motion_pred import *
from train.args_extra_motion import arg_parse
from tools.joint_names import *
from utils.transformations import *
from utils.utils import *

is_train=True
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

class GlobalTrajectoryTrainer:
    def __init__(self, args):
        self.opt = args
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print('Using ', self.opt.gpu_id, ' GPUs!')
        self.num_jts = BODYHAND_JOINTS if self.opt.with_fingers else BODY_JOINTS
        self.frames = self.opt.window_size // self.opt.downsample_rate
        self.batch_size = self.opt.batch_size
        if self.opt.with_fingers:
            dim_pose = MOTION_FEATS_BODYHAND - 3
        else:
            dim_pose =  MOTION_FEATS_BODY_ONLY - 3
    
        # instantiate the model and move it to the right device
        net = Global_Trajectory_Pred( 
            input_feats=dim_pose,    
            output_feats=6,    
            output_emb_width = self.opt.output_emb_width,
            down_t=self.opt.down_t,
            stride_t=self.opt.stride_t,
            width=self.opt.width,
            depth=self.opt.depth,
            dilation_growth_rate=self.opt.dilation_growth_rate,
            norm=self.opt.norm,
            activation=self.opt.act,
            gpu_id=self.opt.gpu_id[0]
            )
        self.device = torch.device('cuda:' + str(self.opt.gpu_id[0]) if torch.cuda.is_available() and sum(self.opt.gpu_id) > -1 else 'cpu')
        print("using device:", self.device)
        self.opt.device = self.device
        self.traj_model = net.to(self.device)
        pc_vq = sum(param.numel() for param in net.parameters())
        print('Total parameters of Trajectory Predictor: {:,}'.format(pc_vq))
        self.opt_traj_model = optim.AdamW(self.traj_model.parameters(), lr=self.opt.lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt_traj_model, step_size=self.opt.step_size, gamma=self.opt.gamma)
        self.epoch = 0
        self.it = 0 
        if self.opt.load_exp is not None:
            print('Loading pre-trained model')
            self.epoch, self.it = self.resume(self.opt.load_exp, change_lr=self.opt.change_lr)
            print('Load model epoch: {}, iterations: {}'.format(self.epoch, self.it))
        
        if self.opt.is_train:
            self.train_loader, self.val_loader = dataset_prepare(self.opt)
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
            self.inv_z_normalization_delta = self.data_loader.dataset.inv_z_normalization_delta
        

    def save(self, dir_name, ep, total_it):
        state = {
            "model": self.traj_model.state_dict(),
            "opt_model": self.opt_traj_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        # Also save the train test loss
        torch.save(state, makepath(os.path.join(dir_name, 'weights.p'), isfile=True))
  

    def resume(self, model_path, change_lr=False):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.traj_model.load_state_dict(checkpoint['model'], strict=True)
        self.opt_traj_model.load_state_dict(checkpoint['opt_model'])
        if change_lr:
            for param_group in self.opt_traj_model.param_groups:
                param_group["lr"] = self.opt.lr
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']


    def forward(self, batch_data, iterations, teacher_forcing):
        loss_model = 0.0
        delta_rec_loss = torch.Tensor([0.0])
        if self.opt.predict_velocity:
            input_delta_motion = torch.cat((batch_data['p0_motion'][..., ROOTTRANS_END_INDEX:].unsqueeze(-1), 
                                            batch_data['p1_motion'][..., ROOTTRANS_END_INDEX:].unsqueeze(-1)), dim=-1).to(self.device)
            output_root_delta = torch.cat((batch_data['p0_motion'][..., :ROOTTRANS_END_INDEX], 
                                            batch_data['p1_motion'][..., :ROOTTRANS_END_INDEX]), dim=-1).to(self.device)
            pred_root_delta = self.traj_model(input_delta_motion)

            delta_rec_loss = self.opt.weight_loss_rec[-1] * self.l1_smoothloss(pred_root_delta, output_root_delta)
            loss_model += delta_rec_loss
            pred_motion_delta = torch.cat((pred_root_delta[..., :ROOTTRANS_END_INDEX],
                                           batch_data['p0_motion'][..., ROOTTRANS_END_INDEX:], 
                                           pred_root_delta[..., ROOTTRANS_END_INDEX:],
                                           batch_data['p1_motion'][..., ROOTTRANS_END_INDEX:]), dim=-1).to(self.device)
            pred_motion_delta_unnormalized = self.inv_z_normalization_delta(pred_motion_delta)
            pred_p0_delta_unnormalized = pred_motion_delta_unnormalized[..., :pred_motion_delta_unnormalized.shape[-1]//2]
            pred_p1_delta_unnormalized = pred_motion_delta_unnormalized[..., pred_motion_delta_unnormalized.shape[-1]//2:]
           
            predict_p0_unnormalized, predict_p1_unnormalized = delta_rel_trans_mot2mot_feats(batch_data['p0_init'], 
                                                                                             pred_p0_delta_unnormalized,
                                                                                             pred_p1_delta_unnormalized)

            pred_motion = torch.cat((predict_p0_unnormalized, predict_p1_unnormalized), dim=-1)
            p0_root_transl, p0_root_rot, p0_body_rot, p0_local_jointpos, p0_local_jointvel, p0_foot_contact = reverse_process_mot(batch_data['p0_motion_unnormalized'], with_fingers=self.opt.with_fingers)
            p1_root_transl, p1_root_rot, p1_body_rot, p1_local_jointpos, p1_local_jointvel, p1_foot_contact = reverse_process_mot(batch_data['p1_motion_unnormalized'], with_fingers=self.opt.with_fingers)
            rel_features = [batch_data['p1_motion_unnormalized'][..., :ROOTTRANS_END_INDEX] - batch_data['p0_motion_unnormalized'][..., :ROOTTRANS_END_INDEX]]
        B, T, dim = pred_motion.shape
        p0_features = [p0_root_transl]
        p1_features = [p1_root_transl]
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
        predict_p0_features = [predict_p0_root_transl]
        predict_p1_features = [predict_p1_root_transl]
        pred_rel_features = [predict_rel_pos]
            
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
        return pred_motion, loss_model, [loss_rec_p0, loss_rec_p1, loss_vel_p0, loss_vel_p1, loss_rel_rec, loss_rel_vel, delta_rec_loss]

    def loss_logs_arrange(self, loss_logs, loss_list):
        loss_logs['loss_rec_p0_root_transl'] += loss_list[0][0].item()
        loss_logs['loss_rec_p1_root_transl'] += loss_list[1][0].item()
        loss_logs['loss_vel_p0_root_transl'] += loss_list[2][0].item()
        loss_logs['loss_vel_p1_root_transl'] += loss_list[3][0].item()
        loss_logs['loss_rel_trans'] += loss_list[4][0].item()
        loss_logs['loss_vel_rel_trans'] += loss_list[5][0].item()
        loss_logs['delta_rec_loss'] += loss_list[6].item()
        return loss_logs
    
    def train_epoch(self, train_loader, epoch, it, teacher_forcing):
        self.traj_model.train()
        train_logs = defaultdict(def_value, OrderedDict())
        mean_loss = defaultdict(def_value, OrderedDict())

        for running_iter, batch_data in enumerate(tqdm(train_loader)):
            self.opt_traj_model.zero_grad()      
            
            pred_motion, loss_model, loss_list = self.forward(batch_data, it, teacher_forcing)
            if loss_model == float('inf') or torch.isnan(loss_model):
                print('Train loss is nan')
                exit()
           
            train_logs['loss'] += loss_model.item()
            train_logs = self.loss_logs_arrange(train_logs, loss_list)
            train_logs['lr'] += self.opt_traj_model.param_groups[0]['lr']
            
            loss_model.mean().backward()
            torch.nn.utils.clip_grad_value_(self.traj_model.parameters(), 0.1)
            self.opt_traj_model.step()
            it += 1
            
        for tag, value in train_logs.items():
            mean_loss[tag] = value / (running_iter + 1)   
        return mean_loss, it

    
    def val_epoch(self, val_loader, epoch, it, teacher_forcing):
        self.traj_model.eval()
        val_logs = defaultdict(def_value, OrderedDict())
        mean_loss = defaultdict(def_value, OrderedDict())
        for running_iter, batch_data in enumerate(tqdm(val_loader)):
            pred_motion, loss_model, loss_list = self.forward(batch_data, it, teacher_forcing)
            
            val_logs['loss'] += loss_model.item()
            val_logs = self.loss_logs_arrange(val_logs, loss_list)
            
            val_logs['lr'] += self.opt_traj_model.param_groups[0]['lr']
        for tag, value in val_logs.items():
            mean_loss[tag] = value / (running_iter + 1)   
        return mean_loss

    def train(self):
        self.traj_model.to(self.device)
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
                                epoch=self.epoch,  lr=self.opt_traj_model.param_groups[0]['lr'])
            if running_epoch % self.opt.eval_every_e == 0:
                start_time_ = time.time()
                val_logs = self.val_epoch(self.val_loader, self.epoch, self.it, teacher_forcing)
                print_current_loss('Val ', os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, self.opt.experiment_name+'_logs.txt'),
                                   time.time() - start_time_, self.it, val_logs['loss'], 
                                   epoch=self.epoch, lr=self.opt_traj_model.param_groups[0]['lr'])
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



if __name__ == "__main__":
    global opt
    opt = arg_parse(is_train=is_train, load_exp=load_exp)
    fixseed(opt.seed)
    trainer = GlobalTrajectoryTrainer(opt)
    if is_train:
        trainer.train()
    