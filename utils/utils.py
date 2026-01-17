import argparse
import json
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import torch.nn.functional as F
from scipy import interpolate
from copy import copy
from PIL import Image
from typing import Any, List
from torch.utils.tensorboard import SummaryWriter

gpu_id = 0   # -1 to use cpu
to_cpu = lambda tensor: tensor.detach().cpu().numpy()
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() and gpu_id >=0 else "cpu")

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model
    
def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].tolist() for k in npz.files}
    return DotDict(npz)

def params2torch(params, dtype = torch.float32, device='cpu'):
    return {k: torch.from_numpy(v).type(dtype).to(device) for k, v in params.items()}

def prepare_params(params, frame_mask, rel_trans = None, dtype = np.float32):
    n_params = {k: v[frame_mask].astype(dtype)  for k, v in params.items()}
    if rel_trans is not None:
        n_params['transl'] -= rel_trans
    return n_params

def torch2np(item, dtype=np.float32):
    out = {}
    for k, v in item.items():
        if v ==[] or v=={}:
            continue
        if isinstance(v, list):
            if isinstance(v[0],  str):
                out[k] = v
            else:
                if torch.is_tensor(v[0]):
                    v = [v[i].cpu() for i in range(len(v))]
                try:
                    out[k] = np.array(np.concatenate(v), dtype=dtype)
                except:
                    out[k] = np.array(np.array(v), dtype=dtype)
        elif isinstance(v, dict):
            out[k] = torch2np(v)
        else:
            if torch.is_tensor(v):
                v = v.cpu()
            out[k] = np.array(v, dtype=dtype) 
            
    return out

def DotDict(in_dict):
    out_dict = copy(in_dict)
    for k,v in out_dict.items():
       if isinstance(v,dict):
           out_dict[k] = DotDict(v)
    return dotdict(out_dict)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def append2dict(source, data):
    for k in data.keys():
        if k in source.keys():
            if isinstance(data[k], list):
                source[k] += data[k]
            else:
                source[k].append(data[k])

            
def append2list(source, data):
    # d = {}
    for k in data.keys():
        leng = len(data[k])
        break
    for id in range(leng):
        d = {}
        for k in data.keys():
            if isinstance(data[k], list):
                if isinstance(data[k][0], str):
                    d[k] = data[k]
                elif isinstance(data[k][0], np.ndarray):
                    d[k] = data[k][id]
                
            elif isinstance(data[k], str):
                    d[k] = data[k]
            elif isinstance(data[k], np.ndarray):
                    d[k] = data[k]
        source.append(d)
           
        # source[k] += data[k].astype(np.float32)
        
        # source[k].append(data[k].astype(np.float32))

def np2torch(item, dtype=torch.float32):
    out = {}
    for k, v in item.items():
        if v ==[] :
            continue
        if isinstance(v, str):
           out[k] = v 
        elif isinstance(v, list):
            # if isinstance(v[0], str):
            #    out[k] = v
            try:
                out[k] = torch.from_numpy(np.concatenate(v)).to(dtype)
            except:
                out[k] = v # torch.from_numpy(np.array(v))
        elif isinstance(v, dict):
            out[k] = np2torch(v)
        else:
            out[k] = torch.from_numpy(v).to(dtype)
    return out

def to_tensor(array, dtype=torch.float32, device='cpu'):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype).to(device)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todencse(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array

def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path

def create_video(path, fps=30,name='movie'):
    import os
    import subprocess
    src = os.path.join(path,'%*.png')
    movie_path = os.path.join(path,'%s.mp4'%name)
    i = 0
    while os.path.isfile(movie_path):
        movie_path = os.path.join(path,'%s_%02d.mp4'%(name,i))
        i +=1

    cmd = 'ffmpeg -f image2 -r %d -i %s -b:v 6400k -pix_fmt yuv420p %s' % (fps, src, movie_path)
    subprocess.call(cmd.split(' '))
    while not os.path.exists(movie_path):
        continue


def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner

@singleton
class _Printer():

    def __init__(self) -> None:
        self._printer = print
        self._debug = False

    def print(self, debug: bool, *args: List[str]) -> None:
        if debug and not self._debug:
            return
        
        args = list(map(str, args))
        self._printer(' '.join(args))

    def setPrinter(self, **kwargs) -> None:
        if 'printer' in kwargs:
            self._printer = kwargs['printer']
        if 'debug' in kwargs:
            self._debug = kwargs['debug']

class Console(object):

    def __init__(self) -> None:
        pass

    @staticmethod
    def setPrinter(**kwargs) -> None:
        """ Set printer for Console

        Args:
            printer: Callable object, core output function
            debug: bool type, whether output debug information 
        """
        p = _Printer()
        p.setPrinter(**kwargs)
    
    @staticmethod
    def log(*args: List[Any]) -> None:
        """ Output log information

        Args:
            args: each element in the input list must be str type
        """
        p = _Printer()
        p.print(False, *args)
    
    @staticmethod
    def debug(*args: List[Any]) -> None:
        """ Output debug information

        Args:
            args: each element in the input list must be str type
        """
        p = _Printer()
        p.print(True, *args)

@singleton
class _Writer():
    def __init__(self) -> None:
        self.writer = None

    def write(self, write_dict: dict) -> None:
        if self.writer is None:
            raise Exception('[ERR-CFG] Writer is None!')
        
        for key in write_dict.keys():
            if write_dict[key]['plot']:
                self.writer.add_scalar(key, write_dict[key]['value'], write_dict[key]['step'])

    def setWriter(self, writer: SummaryWriter) -> None:
        self.writer = writer

class Ploter():
    def __init__(self) -> None:
        pass

    @staticmethod
    def setWriter(writer: SummaryWriter) -> None:
        w = _Writer()
        w.setWriter(writer)
    
    @staticmethod
    def write(write_dict: dict) -> None:
        w = _Writer()
        w.write(write_dict)
        



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

MISSING_VALUE = -1

def save_image(image_numpy, image_path):
    img_pil = Image.fromarray(image_numpy)
    img_pil.save(image_path)


def save_logfile(log_loss, save_path):
    with open(save_path, 'wt') as f:
        for k, v in log_loss.items():
            w_line = k
            for digit in v:
                w_line += ' %.3f' % digit
            f.write(w_line + '\n')


def print_current_loss(split, print_filename, time_elapsed, niter_state, losses, accuracy=None, epoch=None, lr=None, sub_epoch=None,
                       inner_iter=None, tf_ratio=None, sl_steps=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(start_time):
        now = time.time() - start_time
        seconds = int(now % 60)
        return seconds

    if epoch is not None:
        message = split + '- epoch: {:>4d}, niter: {:>6d}, secs/ep: {:.4f}, lr: {:.6f}'.format(epoch, niter_state, time_elapsed, lr)

    # for k, v in losses.items():
    #     message += ' %s: %.4f ' % (k, v)
    message += ' Mean Loss: {:.8f}'.format(losses)
    if accuracy is not None:
        message += ' Acc: {:.5f}'.format(accuracy)
    print(message)
    with open(print_filename, 'a') as f:
        print(message, file=f)


def compose_gif_img_list(img_list, fp_out, duration):
    img, *imgs = [Image.fromarray(np.array(image)) for image in img_list]
    img.save(fp=fp_out, format='GIF', append_images=imgs, optimize=False,
             save_all=True, loop=0, duration=duration)


def save_images(visuals, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = '%d_%s.jpg' % (i, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


def save_images_test(visuals, image_path, from_name, to_name):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = "%s_%s_%s" % (from_name, to_name, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


def compose_and_save_img(img_list, save_dir, img_name, col=4, row=1, img_size=(256, 200)):
    # print(col, row)
    compose_img = compose_image(img_list, col, row, img_size)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_path = os.path.join(save_dir, img_name)
    # print(img_path)
    compose_img.save(img_path)


def compose_image(img_list, col, row, img_size):
    to_image = Image.new('RGB', (col * img_size[0], row * img_size[1]))
    for y in range(0, row):
        for x in range(0, col):
            from_img = Image.fromarray(img_list[y * col + x])
            # print((x * img_size[0], y*img_size[1],
            #                           (x + 1) * img_size[0], (y + 1) * img_size[1]))
            paste_area = (x * img_size[0], y*img_size[1],
                                      (x + 1) * img_size[0], (y + 1) * img_size[1])
            to_image.paste(from_img, paste_area)
            # to_image[y*img_size[1]:(y + 1) * img_size[1], x * img_size[0] :(x + 1) * img_size[0]] = from_img
    return to_image


def plot_loss_curve(losses, save_path, intervals=500):
    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    for key in losses.keys():
        plt.plot(list_cut_average(losses[key], intervals), label=key)
    plt.xlabel("Iterations/" + str(intervals))
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new

def interpolate_by(motion, n):
    T = len(motion)
    T1 = T * n
    x = np.linspace(0, T-1 ,T)
    x_new = np.linspace(0, T-1 ,T1)
    motion_int = np.zeros((T1, motion.shape[-1]))
    for v1 in range(0, motion.shape[-1]):
        p_ = to_np(motion[:, v1])
        f_p = interpolate.interp1d(x, p_, kind = 'linear')
        motion_int[:, v1] = f_p(x_new)
    # motion = to_tensor(motion_int, device=motion.device)
    T = T1
    return motion_int

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_opt(opt_path):
    with open(opt_path) as f: 
        file_lines = f.read()
    args = json.loads(file_lines)
    opt = argparse.Namespace(**args)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    return opt

def normalize_pair_single(pair):
    return tuple(sorted(pair))

def normalize_pair_batch(stacked_tensor):
    # Sort the tensors along the last dimension (the pair dimension)
    sorted_tensors, _ = torch.sort(stacked_tensor, dim=2)
    return sorted_tensors

def add_lookuptable(batch_code0, batch_code1, lookup_table):
    bs = batch_code0.shape[0]
    code_idx_0 = batch_code0.flatten()
    code_idx_1 = batch_code1.flatten()
    lookup_table.update(pair for pair in zip(code_idx_0.tolist(), code_idx_1.tolist()))
    return lookup_table

def normalize_add_lookuptable(batch_code0, batch_code1, lookup_table):
    bs = batch_code0.shape[0]
    code_idx_0 = batch_code0.flatten()
    code_idx_1 = batch_code1.flatten()
    # use normalize_pair_single fuction if you want to keep [A,B]=[B,A] in the lookup table
    lookup_table.update(normalize_pair_single(pair) for pair in zip(code_idx_0.tolist(), code_idx_1.tolist()))
    
    # # Stack the two tensors along a new dimension (dim=2). Don't need this if we use flatten()
    # stacked_batch = torch.stack((batch_code0, batch_code1), dim=2)
    # # Normalize the entire batch without a loop
    # normalized_pair_batch = normalize_pair_batch(stacked_batch)
    # # Convert to a list of tuples by converting each tensor's elements to lists
    # paired_batch = list(zip(normalized_pair_batch[:, :, 0].tolist(), normalized_pair_batch[:, :, 1].tolist()))
    # for batch in range(len(paired_batch)):
    #     lookup_table.update(pair for pair in zip(paired_batch[batch][0], paired_batch[batch][1]))
    return lookup_table

def find_indices(bigger_list, smaller_list):
    # Create a dictionary to map items in bigger_list to their indices
    index_map = {item: idx for idx, item in enumerate(bigger_list)}
    # Use list comprehension to find indices of items in smaller_list
    indices = [index_map.get(item, -1) for item in smaller_list]
    # indices = [index_map[item] for item in smaller_list if item in index_map]
    return indices

def find_indices_partial(lookup_table, pairs):
    indices = []
    for pair in pairs:
        if pair in lookup_table:
            # Exact match
            indices.append(lookup_table.index(pair))
        else:
            # Partial match: find closest pair with at least one matching element
            closest_idx = -1  # Default for no match
            for idx, lookup_pair in enumerate(lookup_table):
                if pair[0] == lookup_pair[0] or pair[1] == lookup_pair[1]:
                    closest_idx = idx
                    break  # Stop at the first partial match
            indices.append(closest_idx)
    return indices

def search_lookuptable(batch_code0, batch_code1, lookup_table, partial_search=True):
    bs = batch_code0.shape[0]
    code_idx_0 = batch_code0.flatten()
    code_idx_1 = batch_code1.flatten()
    pair = [pair for pair in zip(code_idx_0.tolist(), code_idx_1.tolist())]
    if partial_search:
        token_indices = find_indices_partial(lookup_table, pair) 
    else:
        token_indices = find_indices(lookup_table, pair) 
    return to_tensor(token_indices, device=batch_code0.device).int().view(bs, -1)

def normalize_search_lookuptable(batch_code0, batch_code1, lookup_table):
    bs = batch_code0.shape[0]
    code_idx_0 = batch_code0.flatten()
    code_idx_1 = batch_code1.flatten()
    pair = [normalize_pair_single(pair) for pair in zip(code_idx_0.tolist(), code_idx_1.tolist())]
    token_indices = find_indices(lookup_table, pair) 
    return to_tensor(token_indices, device=batch_code0.device).int().view(bs, -1)
    # for pair_idx in pair:
    #     token_index.append(lookup_table.index(pair_idx))

    # # Stack the two tensors along a new dimension (dim=2)
    # stacked_batch = torch.stack((batch_code0, batch_code1), dim=2)
    # # Normalize the entire batch without a loop
    # normalized_pair_batch = normalize_pair_batch(stacked_batch)
    # # Convert to a list of tuples by converting each tensor's elements to lists
    # paired_batch = list(zip(normalized_pair_batch[:, :, 0].tolist(), normalized_pair_batch[:, :, 1].tolist()))
    # combined_token_index = torch.zeros((len(paired_batch)))
    # for batch in range(len(paired_batch)):
    #     combined_token_index[batch] = lookup_table.index(pair for pair in zip(paired_batch[batch][0], paired_batch[batch][1]))
    return token_index

def recover_pair_from_index(index, lookup_table):
    bs = index.shape[0]
    indices = index.flatten().tolist()
    pair = [lookup_table[indices[i]] for i in range(len(indices))]
    code_idx_0, code_idx_1 = zip(*pair)
    return to_tensor(code_idx_0, device=index.device).long().reshape(bs, -1), to_tensor(code_idx_1, device=index.device).long().reshape(bs, -1)


import torch

import torch

def foot_skating_loss(motion_sequence, foot_indices, static_threshold=0.02):
    """
    Computes a foot skating loss for static foot joints by penalizing horizontal movement.

    Args:
        motion_sequence (torch.Tensor): A tensor of shape [T, J, 3] representing the motion sequence.
        foot_indices (list): Indices of the joints representing the feet (e.g., [1, 2] for left and right feet).
        static_threshold (float): Threshold for the Y-axis below which foot joints are considered static.

    Returns:
        torch.Tensor: The foot skating loss value.
    """
    # Extract foot positions for the sequence
    foot_positions = motion_sequence[..., foot_indices, :]  # Shape: [B, T, len(foot_indices), 3]
    
    # Identify static foot joints based on Y-axis threshold
    static_mask = foot_positions[..., 1] < static_threshold  # Shape: [B, T, len(foot_indices)]
    
    # Extract horizontal positions (X and Z) of the foot joints
    foot_positions_horizontal = foot_positions[..., [0, 2]]  # Shape: [B, T, len(foot_indices), 2]
    
    # Compute velocities (differences between consecutive frames)
    foot_velocities = torch.diff(foot_positions_horizontal, dim=1)  # Shape: [B, T-1, len(foot_indices), 2]
    
    # Apply the static mask to velocities (set velocities to 0 where the foot is static)
    static_mask_shifted = static_mask[:, :-1, :]  # Align mask with velocity frames
    foot_velocities[static_mask_shifted] = 0  # Zero out velocities for static frames
    
    # Compute the skating loss as the L2 norm of the horizontal velocities
    skating_loss = torch.sum(torch.norm(foot_velocities, dim=-1))  # Sum over all frames and foot joints
    
    return skating_loss


def adjust_foot_sliding(motion_sequence, foot_indices, ground_threshold=0.05, slide_threshold=1.2):
    """
    Detects and adjusts foot sliding in ground-contact segments by setting foot joints to points above the ground
    proportional to the horizontal distance covered.

    Args:
        motion_sequence (torch.Tensor): A tensor of shape [T, J, 3] representing the motion sequence.
        foot_indices (list): Indices of the joints representing the feet (e.g., [1, 2] for left and right feet).
        ground_threshold (float): Threshold for determining ground contact based on vertical position.
        slide_threshold (float): Horizontal distance threshold to identify foot sliding.

    Returns:
        torch.Tensor: Adjusted motion sequence with foot sliding corrected.
    """
    T, J, _ = motion_sequence.shape

    # Extract the vertical positions (Y-axis) of the foot joints
    foot_positions = motion_sequence[:, foot_indices, :]  # Shape: [T, len(foot_indices), 3]

    # Check if all foot joints are within the ground threshold
    on_ground = torch.all(torch.abs(foot_positions[:, :, 1]) < ground_threshold, dim=1)  # Shape: [T]

    # Find contiguous segments where `on_ground` is True
    segments = []
    start = None
    for t in range(T):
        if on_ground[t]:
            if start is None:
                start = t  # Start a new segment
        else:
            if start is not None:
                segments.append((start, t - 1))  # End the current segment
                start = None
    if start is not None:  # Handle the last segment if it ends at the final frame
        segments.append((start, T - 1))

    # Adjust motion sequence for foot sliding
    adjusted_motion_sequence = motion_sequence.clone()
    for start, end in segments:
        for i, foot_index in enumerate(foot_indices):
            # Extract horizontal positions (X, Z) of the foot in the segment
            foot_positions_segment = foot_positions[start:end + 1, i, :][:, [0, 2]]  # X and Z coordinates
            # Compute differences between consecutive frames (velocities)
            foot_velocities = torch.diff(foot_positions_segment, dim=0)
            # Compute the horizontal distances (L2 norm of velocities)
            horizontal_distances = torch.norm(foot_velocities, dim=1)

            # Adjust frames with foot sliding
            for frame_idx in range(start + 1, end + 1):  # Skip the first frame
                if horizontal_distances[frame_idx - start - 1] > slide_threshold:
                    distance = horizontal_distances[frame_idx - start - 1].item()
                    # Set a new vertical position proportional to the horizontal distance
                    adjusted_height = ground_threshold + 0.1 * distance  # Example scaling
                    adjusted_motion_sequence[frame_idx, foot_index, 1] = adjusted_height

    return adjusted_motion_sequence

def save_smplx_as_npz(data_dict_, filename):
    data = {
            'gender': data_dict_['meta']['gender'],
            'surface_model_type': 'smplx',
            'mocap_frame_rate': 30,
            # 'mocap_time_length': 20,
            'trans':np.array(data_dict_['transl'][:800], dtype=np.float32),
            'poses':np.array(data_dict_['poses'][:800], dtype=np.float32),
            'betas':np.array(data_dict_['betas'][0], dtype=np.float32),
            'root_orient':np.array(data_dict_['global_orient'][:800], dtype=np.float32),
            'pose_body':np.array(data_dict_['poses'][:800, 3:66], dtype=np.float32),
        }
    np.savez(filename, **data)