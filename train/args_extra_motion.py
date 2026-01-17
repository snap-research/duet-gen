import argparse
import os
import json
import torch
import time

## dataloader
dataset_name = 'DD100'
window_size = 400
window_stride = 100
downsample_rate = 2
data_scale = 1.0
with_fingers = False
datafolder = 'fingers' if with_fingers else 'no_fingers'
dataset_root = os.path.join('data', dataset_name, str(window_size) + '_' + str(window_stride))
checkpoints_dir = os.path.join('checkpoints', dataset_name, 'extra_motion')
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
num_genres = 10
predict_velocity = True
## architecture hyper parameters
model_name = 'Global_Trajectory_Pred'
output_emb_width = 512

down_t = 2
stride_t = 2
width = 512
depth = 3
dilation_growth_rate = 3
mu = 0.99
norm = 'LN'
act = 'silu'
quantizer = "ema_reset"
num_quantizers = 4
quantize_dropout_prob = 0.2
vq_pretrained_weight_path =  '' 

## train & test
gpu_id = [0]
batch_size = 256
seed = 3407
learning_rate = 1e-4
change_lr=True
num_epoch = 5500
use_cuda = True
resume_model = ''
num_workers = 0
total_iter = 30000
milestones = [150000, 250000]
step_size = 100
gamma = 0.999
warm_up_iter = 2000
eval_every_e = 5
save_every_e = 25
log_every_e = 5

## training losses
weight_loss_rec = [1.0, 1.0, 2.0, 1.0, 10.0, 1.0, 1.0, 2.0, 2.0, 2.0]
weight_loss_vel = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
weight_loss_relative_pos = [1.0, 1.0, 5.0, 5.0]
weight_loss_relative_vel = [10.0, 10.0, 10.0, 10.0]
weight_loss_foot_contact = 1.0
weight_loss_commit = 0.1
weight_loss_rec_vertex = 1.0
weight_loss_kl = 0.1
weight_loss_fk = 0.01
weight_loss_vposer = 1e-3
weight_loss_ground = 1.0


def arg_parse(is_train=False, load_exp=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--dataset_name', type=str, default=dataset_name, help='dataset directory')
    parser.add_argument('--num_genres', type=str, default=num_genres, help='dance genres in dataset')
    parser.add_argument('--dataset_root', type=str, default=dataset_root, help='dataset directory')
    parser.add_argument('--window_size', type=int, default=window_size, help='training motion length')
    parser.add_argument('--downsample_rate', type=int, default=downsample_rate, help='downsample rate of each mini sequence')
    parser.add_argument('--data_scale', type=int, default=data_scale, help='scale down data before training')
    parser.add_argument("--with_fingers", type=bool, default=with_fingers, help='Training with finger joints?')
    parser.add_argument("--predict_velocity", type=bool, default=predict_velocity, help='Predict root velocities instead of position?')
    parser.add_argument("--gpu_id", type=int, default=gpu_id, help='GPU id')
    parser.add_argument('--checkpoints_dir', type=str, default=checkpoints_dir, help='models are saved here')
    ## path setting
    parser.add_argument('--logs_dir', 
                        type=str, 
                        default='/logs',
                        help='dir for saving checkpoints and logs')
    parser.add_argument('--stamp', 
                        type=str, 
                        default=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
                        help='timestamp')
    parser.add_argument('--vq_pretrained_weight_path', type=str, default=vq_pretrained_weight_path, help='Pre-trained VQ model')
    
    ## architecture
    parser.add_argument('--model_name', type=str, default=model_name, help='Name of this model')
    parser.add_argument("--down_t", type=int, default=down_t, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=stride_t, help="stride size")
    parser.add_argument("--width", type=int, default=width, help="width of the network")
    parser.add_argument("--depth", type=int, default=depth, help="num of resblocks for each res")
    parser.add_argument("--dilation_growth_rate", type=int, default=dilation_growth_rate, help="dilation growth rate")
    parser.add_argument("--output_emb_width", type=int, default=output_emb_width, help="output embedding width")
    parser.add_argument('--act', type=str, default=act, choices=['relu', 'silu', 'gelu'],
                        help='activation function')
    parser.add_argument('--norm', type=str, default=norm, choices=['LN', 'GN', 'BN'],
                         help='layer norm / group norm / batch norm inside ResNet')

 
    
    
    ## training and optimization
    parser.add_argument('--experiment_name', type=str, default=None, help='Name of this trial')
    parser.add_argument("--seed", default=seed, type=int)
    parser.add_argument('--total_iter', default=total_iter, type=int, help='number of total iterations to run')
    parser.add_argument('--warm_up_iter', default=warm_up_iter, type=int, help='number of total iterations for warmup')
    
    parser.add_argument('--milestones', default=milestones, nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--step_size', default=step_size, nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=gamma, type=float, help="learning rate decay")

    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=batch_size,
                        help='batch size to train')
    parser.add_argument('--lr', 
                        type=float, 
                        default=learning_rate,
                        help='initial learing rate')
    parser.add_argument('--change_lr', 
                        type=bool, 
                        default=change_lr,
                        help='initial learing rate')
    parser.add_argument('--num_epoch', 
                        type=int, 
                        default=num_epoch,
                        help='#epochs to train')
    parser.add_argument('--use_cuda', 
                        type=str, 
                        default=use_cuda,
                        help='set device for training')
    parser.add_argument('--resume_model',
                        type=str,
                        default=resume_model,
                        help='resume model path')
    parser.add_argument('--num_workers',
                        type=int,
                        default=num_workers,
                        help='number of dataloader worker processer')
    parser.add_argument('--is_continue', action="store_true", help='Name of this trial')

    parser.add_argument('--log_every_e', default=log_every_e, type=int, help='iter log frequency')
    parser.add_argument('--save_latest', default=500, type=int, help='iter save latest model frequency')
    parser.add_argument('--save_every_e', default=save_every_e, type=int, help='save model every n epoch')
    parser.add_argument('--eval_every_e', default=eval_every_e, type=int, help='save eval results every n epoch')
    parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')

    parser.add_argument('--which_epoch', type=str, default="all", help='Name of this trial')
    
    
    ## training loss
    parser.add_argument('--weight_loss_commit', type=float, default=weight_loss_commit, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')
    parser.add_argument('--weight_loss_rec',
                        type=list,
                        default=weight_loss_rec,
                        help='loss weight of rec loss')
    parser.add_argument('--weight_loss_vel',
                        type=list,
                        default=weight_loss_vel,
                        help='loss weight of velocity loss')
    parser.add_argument('--weight_loss_foot_contact',
                        type=float,
                        default=weight_loss_foot_contact,
                        help='loss weight of binary cross entropy loss for foot contacts')
    parser.add_argument('--weight_loss_relative_pos',
                        type=list,
                        default=weight_loss_relative_pos,
                        help='loss weight of relative distance between two bodies')
    parser.add_argument('--weight_loss_relative_vel',
                        type=list,
                        default=weight_loss_relative_vel,
                        help='loss weight of relative velocity between two bodies')
    parser.add_argument('--weight_loss_kl',
                        type=float,
                        default=weight_loss_kl,
                        help='loss weight of kl loss')
    parser.add_argument('--weight_loss_vposer',
                        type=float,
                        default=weight_loss_vposer,
                        help='loss weight of vposer loss')
    parser.add_argument('--weight_loss_ground',
                        type=float,
                        default=weight_loss_ground,
                        help='loss weight of ground loss')
    parser.add_argument('--all_body_vertices',
                        action="store_true",
                        help='use all body vertices to regress')
    
  

    if load_exp is None:
        if is_train:
            opt = parser.parse_args()
            opt.is_train = is_train
            opt.load_exp = load_exp
            # create new experiment
            opt.experiment = _update_exp()
            if with_fingers == True:
                with_f =  'fingers'
            else:
                with_f =  'nofingers'
            experiment_name = 'exp_' + str(opt.experiment) + '_' + opt.model_name + '_' + str(opt.batch_size) + '_' + str(opt.window_size) + '_' + with_f
            opt.experiment_name = experiment_name
            expr_dir = os.path.join(opt.checkpoints_dir, experiment_name)
            opt.logs_dir = os.path.join(expr_dir, 'tenserboard_logs')
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir)
            if not os.path.exists(opt.logs_dir):
                os.makedirs(opt.logs_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            args = vars(opt)
            json.dump(args, open(file_name,'w'))
            # with open(file_name, 'wt') as opt_file:
            #     for k, v in sorted(args.items()):
            #         opt_file.write('%s: %s\n' % (str(k), str(v)))
        
    else:  
        assert load_exp.endswith('.p')
        expr_dir = os.path.dirname(os.path.dirname(load_exp))
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name) as f: 
            file_lines = f.read()
        args = json.loads(file_lines)
        args['load_exp'] = load_exp
        args['is_train'] = is_train
        opt = argparse.Namespace(**args)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
    return opt

def _update_exp():
    exp_file = '.experiments'
    if not os.path.exists(exp_file):
        exp = -1
        with open(exp_file, 'w') as f:
            f.writelines([f'{exp}\n'])
    else:
        with open(exp_file, 'r') as f:
            lines = f.readlines()
        exp = int(lines[0].strip())
    exp += 1
    with open(exp_file, 'w') as f:
        f.writelines([f'{exp}\n'])
    print(f'Experiment Number: {exp}')
    return exp