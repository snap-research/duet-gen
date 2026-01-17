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
checkpoints_dir = os.path.join('checkpoints', dataset_name, 'masked_transformer')
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
num_genres = 10
predict_velocity = True
## VQVae architecture hyper parameters

vq_pretrained_weight_path = os.path.join('checkpoints', dataset_name, 'vqvae',
                                          'exp_5_Hierarchical_VQVAE_TwoPerson_64_400_nofingers', '4175', 'weights.p')
                                        
topcode_transformer_weight_path = os.path.join('checkpoints', 'DD100', 'masked_transformer', 'exp_15_Music2Motion_combined_256_400_nofingers', '1295', 'weights.p')
code_length = 50
temperature = 1
time_steps = 20
topkr = 0.9
cond_scale = 4
# Masked transformer architecture hyperparameters
model_name = 'Music2Motion_combined_hier_bottom' 
genre_emb_dim = 50
p_emb_dim = 128
latent_dim = 512
music_feats_dim = 120
cond_drop_prob = 0.1
ff_size = 1024
num_layers = 8
num_heads = 8
dropout = 0.2
activation = 'gelu'
noise_schedule = 'cosine'
norm = 'LN'
## train & test
gpu_id = [0]
batch_size = 128
seed = 3407
learning_rate = 2e-5
change_lr=True
num_epoch = 3500
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
weight_loss_rec = [10.0, 10.0, 2.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0]
weight_loss_vel = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
weight_loss_relative_pos = [2.0, 2.0, 2.0, 1.0]
weight_loss_relative_vel = [2.0, 2.0, 2.0, 1.0]
weight_loss_foot_contact = 1.0
weight_ce_loss = 1.0
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

    ## vqvae pre-trained model
    parser.add_argument('--vq_pretrained_weight_path', type=str, default=vq_pretrained_weight_path, help='Pre-trained VQ model')
    parser.add_argument('--topcode_transformer_weight_path', type=str, default=topcode_transformer_weight_path, 
                        help='Pre-trained topcode transformer model required to train the bottomcode transformer')
    parser.add_argument('--code_length', type=int, default=code_length, help='Predicted token length during inference.')

    ## transformer architecture hyperparameters
    parser.add_argument('--model_name', type=str, default=model_name, help='Name of this model')
    parser.add_argument('--latent_dim', type=int, default=latent_dim, help='Dimension of transformer latent.')
    parser.add_argument("--genre_emb_dim", type=int, default=genre_emb_dim, help="embedding dim of genre conditions")
    parser.add_argument("--p_emb_dim", type=int, default=p_emb_dim, help="embedding dim of num person conditions")
    parser.add_argument('--music_feats_dim', type=int, default=music_feats_dim, help='Dimension of music features.')
    parser.add_argument('--n_heads', type=int, default=num_heads, help='Number of heads.')
    parser.add_argument('--n_layers', type=int, default=num_layers, help='Number of attention layers.')
    parser.add_argument('--ff_size', type=int, default=ff_size, help='FF_Size')
    parser.add_argument('--dropout', type=float, default=dropout, help='Dropout ratio in transformer')
    parser.add_argument('--cond_drop_prob', type=float, default=cond_drop_prob, help='Drop ratio of condition, for classifier-free guidance')
    parser.add_argument('--activation', type=str, default=activation, choices=['relu', 'silu', 'gelu'],
                        help='activation function')
    parser.add_argument('--noise_schedule', type=str, default=noise_schedule, choices=['cosine', 'scaled_cosine', 'linear', 'q'],
                        help='noise scheduling function for masking tokens')
    parser.add_argument('--norm', type=str, default=norm, choices=['LN', 'GN', 'BN'],
                         help='layer norm / group norm / batch norm')

    
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
    parser.add_argument('--weight_ce_loss',
                        type=float,
                        default=weight_ce_loss,
                        help='loss weight of cross entropy loss')
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
    ## inference time 
    parser.add_argument("--repeat_times", default=1, type=int,
                                help="Number of repetitions, per sample text prompt")
    parser.add_argument("--cond_scale", default=cond_scale, type=float,
                                help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    parser.add_argument("--temperature", default=temperature, type=float,
                                help="Sampling Temperature.")
    parser.add_argument("--topkr", default=topkr, type=float,
                                help="Filter out percentil low prop entries.")
    parser.add_argument("--time_steps", default=time_steps, type=int,
                                help="Mask Generate steps.")
    parser.add_argument('--gumbel_sample', action="store_true", help='True: gumbel sampling, False: categorical sampling.')
        
    assert vq_pretrained_weight_path is not None
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