import os
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
from collections import OrderedDict, defaultdict
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.transformations import *
from dataset_preparation.dd100_loader import DD100_dataset_3 as DD100_dataset
from dataset_preparation.data_utils import *
from models.vqvae.motion_vqvae import *
from models.transformer.music2motion import *
from train.args_masked_transformer_bottom import arg_parse
from utils.utils import *

is_train=True
test_split = 'test'  # the dataset split that you want to evaluate on.
load_exp = None
do_visualize = False

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



class MaskedTransformerTrainer:
    def __init__(self, args):
        self.opt = args
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print('Using ', self.opt.gpu_id, ' GPUs!')
        
        if self.opt.with_fingers:
            self.input_dim_pose = (2 * MOTION_FEATS_BODYHAND)
        else:
            self.input_dim_pose = (2 * MOTION_FEATS_BODY_ONLY)
        self.num_jts = BODYHAND_JOINTS if self.opt.with_fingers else BODY_JOINTS
        self.vq_model, self.vq_opt, self.topcode_transformer_model, self.topcode_trans_opt = self.load_pretrained_models(self.opt.vq_pretrained_weight_path, 
                                                                                                                    self.opt.topcode_transformer_weight_path)
        
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
        self.opt_model = optim.AdamW(self.transformer_model.parameters(), lr=self.opt.lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt_model, step_size=self.opt.step_size, gamma=self.opt.gamma)
        self.epoch = 0
        self.it = 0 
        if self.opt.load_exp is not None:
            print('Loading pre-trained model')
            self.epoch, self.it = self.resume(self.opt.load_exp, change_lr=self.opt.change_lr)
            print('Load model epoch: {}, iterations: {}'.format(self.epoch, self.it))
        
        
        if self.opt.is_train:
            self.train_loader, self.val_loader = dataset_prepare(self.opt)
            self.z_normalization_delta = self.train_loader.dataset.z_normalization_delta
            self.inv_z_normalization_delta = self.train_loader.dataset.inv_z_normalization_delta
            self.logger = SummaryWriter(log_dir=os.path.join(self.opt.logs_dir, str(datetime.now()).replace(" ", "_")))
        else:
            self.load_dataset = DD100_dataset(data_root=self.opt.dataset_root, split=test_split, window_size=self.opt.window_size, 
                                     with_fingers=self.opt.with_fingers, predict_velocity=self.opt.predict_velocity,
                                     data_scale=self.opt.data_scale,
                                     downsample_rate=self.opt.downsample_rate, device=self.opt.device)
            self.data_loader = DataLoader(self.load_dataset, batch_size=1, pin_memory=False,
                                        drop_last=True, num_workers=self.opt.num_workers,
                                        shuffle=False)
            self.z_normalization_delta = self.data_loader.dataset.z_normalization_delta
            self.inv_z_normalization_delta = self.data_loader.dataset.inv_z_normalization_delta
            

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
 

    def save(self, dir_name, ep, total_it):
        state = {
            "vq_model": self.transformer_model.state_dict(),
            "opt_vq_model": self.opt_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        # Also save the train test loss
        torch.save(state, makepath(os.path.join(dir_name, 'weights.p'), isfile=True))
  

    def resume(self, model_path, change_lr=False):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.transformer_model.load_state_dict(checkpoint['vq_model'], strict=True)
        self.opt_model.load_state_dict(checkpoint['opt_vq_model'])
        if change_lr:
            for param_group in self.opt_model.param_groups:
                param_group["lr"] = self.opt.lr
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']
   

    def forward(self, batch_data, iterations, teacher_forcing):
        loss_model = 0.0
        delta_rec_loss = torch.Tensor([0.0])
        if self.opt.predict_velocity:
            input_delta_motion = torch.cat((batch_data['p0_motion'], batch_data['p1_motion']), dim=-1).to(self.device)
            code_idx = self.vq_model.encode(input_delta_motion)
            randomizer = torch.randint(0, 4, (1, 1))
            if randomizer == 1:
                # generating hierarchical top codes and replacing them in the GT (noisy inputs)
                pred_top_code_idx, ce_loss_top, acc_top  = self.topcode_transformer_model(code_idx[1],
                                                            batch_data['music_feats_mfcc'],
                                                            batch_data['music_feats_chroma'],
                                                            iteration=15000
                                                            )

                code_idx = [code_idx[0], pred_top_code_idx]

                                                         
            pred_code, ce_loss, acc  = self.transformer_model(code_idx,
                                                            batch_data['music_feats_mfcc'],
                                                            batch_data['music_feats_chroma'],
                                                            iterations
                                                            )
            loss_model += ce_loss
            # loss_model += mean_proximity_loss

        return loss_model, [ce_loss], acc

    def loss_logs_arrange(self, loss_logs, loss_list):

        loss_logs['ce_loss'] += loss_list[0].item()
        # loss_logs['mean_prox_loss'] += loss_list[1].item()
        return loss_logs
     
    def train_epoch(self, train_loader, epoch, it, teacher_forcing):
        self.transformer_model.train()
        self.topcode_transformer_model.eval()
        self.vq_model.eval()
        train_logs = defaultdict(def_value, OrderedDict())
        mean_loss = defaultdict(def_value, OrderedDict())
        for running_iter, batch_data in enumerate(tqdm(train_loader)):
            self.opt_model.zero_grad()       
            loss_model, loss_list, acc = self.forward(batch_data, it, teacher_forcing)
            if loss_model == float('inf') or torch.isnan(loss_model):
                print('Train loss is nan')
                exit()
            train_logs['loss'] += loss_model.item()
            train_logs = self.loss_logs_arrange(train_logs, loss_list)
            train_logs['acc'] += acc
            train_logs['lr'] += self.opt_model.param_groups[0]['lr']
            loss_model.mean().backward()
            torch.nn.utils.clip_grad_value_(self.transformer_model.parameters(), 0.1)
            self.opt_model.step()
            it += 1
        for tag, value in train_logs.items():
            mean_loss[tag] = value / (running_iter + 1)   
        return mean_loss, it

    def val_epoch(self, val_loader, epoch, it, teacher_forcing):
        self.transformer_model.eval()
        self.topcode_transformer_model.eval()
        self.vq_model.eval()
        val_logs = defaultdict(def_value, OrderedDict())
        mean_loss = defaultdict(def_value, OrderedDict())
        
        for running_iter, batch_data in enumerate(val_loader):

            loss_model, loss_list, acc = self.forward(batch_data, it, teacher_forcing)
            val_logs['loss'] += loss_model.item()
            val_logs = self.loss_logs_arrange(val_logs, loss_list)
            val_logs['acc'] += acc
            val_logs['epoch'] += self.opt_model.param_groups[0]['lr']
        for tag, value in val_logs.items():
            mean_loss[tag] = value / (running_iter + 1)   
        return mean_loss

    def train(self):
        self.transformer_model.to(self.device)
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
                               time.time() - start_time, self.it, train_logs['loss'], accuracy=train_logs['acc'],
                                epoch=self.epoch,  lr=self.opt_model.param_groups[0]['lr'])
            if running_epoch % self.opt.eval_every_e == 0:
                start_time_ = time.time()
                val_logs = self.val_epoch(self.val_loader, self.epoch, self.it, teacher_forcing)
                print_current_loss('Val ', os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name, self.opt.experiment_name+'_logs.txt'),
                                   time.time() - start_time_, self.it, val_logs['loss'], accuracy=val_logs['acc'],
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

    def batch_visualize_predicted(self, batch_data, pred_motion, count, v, do_visualize=True):
        if self.opt.predict_velocity:
            gt_p0 = batch_data['p0_motion_unnormalized']
            gt_p1 = batch_data['p1_motion_unnormalized']
            unnormalized_pred_motion = pred_motion

        predict_p0 = unnormalized_pred_motion[..., :unnormalized_pred_motion.shape[-1]//2]
        predict_p1 = unnormalized_pred_motion[..., unnormalized_pred_motion.shape[-1]//2:]
        data0 = mot_to_smplx(gt_p0[0], betas=batch_data['p0_betas'][0], gender='male' if batch_data['p0_gender'][0] == 1 else 'female',
                             with_fingers=self.opt.with_fingers, interpolate_by=self.opt.downsample_rate)
        pred_data0 = mot_to_smplx(predict_p0[0], betas=batch_data['p0_betas'][0], gender='male' if batch_data['p0_gender'][0] == 1 else 'female',
                             with_fingers=self.opt.with_fingers, interpolate_by=self.opt.downsample_rate)
        data1 = mot_to_smplx(gt_p1[0], betas=batch_data['p1_betas'][0], gender='male' if batch_data['p1_gender'][0] == 1 else 'female',
                             with_fingers=self.opt.with_fingers, interpolate_by=self.opt.downsample_rate)
        pred_data1 = mot_to_smplx(predict_p1[0], betas=batch_data['p1_betas'][0], gender='male' if batch_data['p1_gender'][0] == 1 else 'female',
                             with_fingers=self.opt.with_fingers, interpolate_by=self.opt.downsample_rate)
        if do_visualize:
            from tools.visualize_data import visualize_sequence
            visualize_sequence(data0, data1, pred_data0, pred_data1)
        else:
            from tools.visualize_data import save_sequence
            savepath = makepath(os.path.join(os.path.dirname(self.opt.load_exp), test_split, str(count) + '.mp4'), isfile=True)
            save_sequence(v, savepath, pred_data0, pred_data1)
            
            
    def test(self): 
        from aitviewer.headless import HeadlessRenderer
        v = HeadlessRenderer()
        self.transformer_model.eval()
        self.topcode_transformer_model.eval()
        self.vq_model.eval()
        for running_iter, batch_data in enumerate(self.data_loader):
            
            input_delta_motion = torch.cat((batch_data['p0_motion'], batch_data['p1_motion']), dim=-1).to(self.device)
            code_idx = self.vq_model.encode(input_delta_motion)
            pred_top_code_idx  = self.topcode_transformer_model.generate(batch_data['music_feats_mfcc'],
                                                                    batch_data['music_feats_chroma'],
                                                                    token_len=25,
                                                                    timesteps=20
                                                                    )
            
            pred_bottom_code_idx  = self.transformer_model.generate(pred_top_code_idx,
                                                            batch_data['music_feats_mfcc'],
                                                            batch_data['music_feats_chroma'],
                                                            token_len=50,
                                                            timesteps=20
                                                            )
            pred_code_idx = [pred_bottom_code_idx, pred_top_code_idx]
            pred_motion_delta = self.vq_model.decode(pred_code_idx, batch_data['genre_class'], batch_data['music_feats_mfcc'], batch_data['music_feats_chroma'])
            
            # inv_z_normalize the predicted deltas and add it with the inits
            pred_motion_delta_unnormalized = self.inv_z_normalization_delta(pred_motion_delta)
            pred_p0_delta_unnormalized = pred_motion_delta_unnormalized[..., :pred_motion_delta_unnormalized.shape[-1]//2]
            pred_p1_delta_unnormalized = pred_motion_delta_unnormalized[..., pred_motion_delta_unnormalized.shape[-1]//2:]
            predict_p0_unnormalized, predict_p1_unnormalized = delta_rel_trans_mot2mot_feats(batch_data['p0_init'], 
                                                                                             pred_p0_delta_unnormalized,
                                                                                             pred_p1_delta_unnormalized)
            pred_motion = torch.cat((predict_p0_unnormalized, predict_p1_unnormalized), dim=-1)
            
            # quantitative evaluations here.
            self.batch_visualize_predicted(batch_data, pred_motion, running_iter, v, do_visualize=do_visualize)

if __name__ == "__main__":
    global opt
    opt = arg_parse(is_train=is_train, load_exp=load_exp)
    fixseed(opt.seed)
    
    trainer = MaskedTransformerTrainer(opt)
    if is_train:
        trainer.train()
    else:
        trainer.test()