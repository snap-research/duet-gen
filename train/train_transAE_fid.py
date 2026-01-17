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
from torch.nn.utils import clip_grad_norm_
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

class Trainer:
    def __init__(self, args):
        self.opt = args
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print('Using ', self.opt.gpu_id, ' GPUs!')
        self.num_jts = BODYHAND_JOINTS if self.opt.with_fingers else BODY_JOINTS
        self.frames = self.opt.window_size // self.opt.downsample_rate
        self.batch_size = self.opt.batch_size
        dim_pose = self.num_jts * 3
        
        # instantiate the model and move it to the right device
        net = TransAE(nfeats=dim_pose)
        self.device = torch.device('cuda:' + str(self.opt.gpu_id[0]) if torch.cuda.is_available() and sum(self.opt.gpu_id) > -1 else 'cpu')
        print("using device:", self.device)
        self.opt.device = self.device
        self.transAE_model = net.to(self.device)
        pc_ae = sum(param.numel() for param in net.parameters())
        print('Total parameters of transAE: {:,}'.format(pc_ae))
        self.opt_model = optim.AdamW(self.transAE_model.parameters(), lr=self.opt.lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt_model, step_size=self.opt.step_size, gamma=self.opt.gamma)
        self.epoch = 0
        self.it = 0 
        if self.opt.load_exp is not None:
            print('Loading pre-trained model')
            self.epoch, self.it = self.resume(self.opt.load_exp, change_lr=self.opt.change_lr)
            print('Load model epoch: {}, iterations: {}'.format(self.epoch, self.it))
        
        if self.opt.is_train:
            self.train_loader, self.val_loader = dataset_prepare(self.opt)
            self.logger = SummaryWriter(log_dir=os.path.join(self.opt.logs_dir, str(datetime.now()).replace(" ", "_")))
            self.l1_smoothloss = torch.nn.SmoothL1Loss()
        
        else:
            self.load_dataset = DD100_dataset(data_root=self.opt.dataset_root, split=test_split, window_size=self.opt.window_size, 
                                     with_fingers=self.opt.with_fingers, predict_velocity=self.opt.predict_velocity,
                                     data_scale=self.opt.data_scale,
                                     downsample_rate=self.opt.downsample_rate, device=self.opt.device)
            self.data_loader = DataLoader(self.load_dataset, batch_size=1, pin_memory=False,
                                        drop_last=False, num_workers=self.opt.num_workers,
                                        shuffle=False)
            
    def save(self, dir_name, ep, total_it):
        state = {
            "model": self.transAE_model.state_dict(),
            "opt_model": self.opt_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        # Also save the train test loss
        torch.save(state, makepath(os.path.join(dir_name, 'weights.p'), isfile=True))
  

    def resume(self, model_path, change_lr=False):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.transAE_model.load_state_dict(checkpoint['model'], strict=True)
        self.opt_model.load_state_dict(checkpoint['opt_model'])
        if change_lr:
            for param_group in self.opt_model.param_groups:
                param_group["lr"] = self.opt.lr
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']


    def forward(self, batch_data, iterations, teacher_forcing):
        p0_globalpose = batch_data['p0_globalpos'][:, :, :self.num_jts]
        p0_globalpose = p0_globalpose.reshape(p0_globalpose.shape[0], p0_globalpose.shape[1], -1)
        # For single person FID training
        pred_motion_delta = self.transAE_model(p0_globalpose)
        rec_loss = self.opt.weight_loss_rec[-1] * self.l1_smoothloss(pred_motion_delta, p0_globalpose)
        velocity_loss = self.opt.weight_loss_vel[-1] * self.l1_smoothloss(pred_motion_delta[:, 1:] - pred_motion_delta[:, :-1],
                                                                         p0_globalpose[:, 1:] - p0_globalpose[:, :-1])
        loss_model = rec_loss + velocity_loss
        
        return loss_model

    
    def train_epoch(self, train_loader, epoch, it, teacher_forcing):
        self.transAE_model.train()
        train_logs = defaultdict(def_value, OrderedDict())
        mean_loss = defaultdict(def_value, OrderedDict())

        for running_iter, batch_data in enumerate(tqdm(train_loader)):
            self.opt_model.zero_grad()      
            
            loss_model = self.forward(batch_data, it, teacher_forcing)
            if loss_model == float('inf') or torch.isnan(loss_model):
                print('Train loss is nan')
                exit()
           
            train_logs['loss'] += loss_model.item()
            train_logs['lr'] += self.opt_model.param_groups[0]['lr']
            
            loss_model.mean().backward()
            torch.nn.utils.clip_grad_value_(self.transAE_model.parameters(), 0.1)
            self.opt_model.step()
            it += 1
            
        for tag, value in train_logs.items():
            mean_loss[tag] = value / (running_iter + 1)   
        return mean_loss, it

    
    def val_epoch(self, val_loader, epoch, it, teacher_forcing):
        self.transAE_model.eval()
        val_logs = defaultdict(def_value, OrderedDict())
        mean_loss = defaultdict(def_value, OrderedDict())
        for running_iter, batch_data in enumerate(tqdm(val_loader)):
            loss_model = self.forward(batch_data, it, teacher_forcing)
            
            val_logs['loss'] += loss_model.item()
            val_logs['epoch'] += self.opt_model.param_groups[0]['lr']
            

        for tag, value in val_logs.items():
            mean_loss[tag] = value / (running_iter + 1)   
        return mean_loss

    def train(self):
        self.transAE_model.to(self.device)
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
                                epoch=self.epoch,  lr=self.opt_model.param_groups[0]['lr'])
            if running_epoch % self.opt.eval_every_e == 0:
                start_time_ = time.time()
                val_logs = self.val_epoch(self.val_loader, self.epoch, self.it, teacher_forcing)
                print_current_loss('Val ', os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, self.opt.experiment_name+'_logs.txt'),
                                   time.time() - start_time_, self.it, val_logs['loss'], 
                                   epoch=self.epoch, lr=self.opt_model.param_groups[0]['lr'])
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
    generate_lookup_table = False
    is_evaluate = True
    trainer = Trainer(opt)
    trainer.train()