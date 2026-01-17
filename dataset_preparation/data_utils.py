import torch
from smplx import SMPLX
from scipy import interpolate
from tools.joint_names import *
from utils.quaternion import *
from utils.transformations import *
from utils.utils import *

BODYHAND_JOINTS = 55
BODY_JOINTS = 22
HAND_JOINTS = 15
ROOTTRANS_END_INDEX = 3
ROOTROT_END_INDEX = ROOTTRANS_END_INDEX + 6
BODYROT_END_INDEX = ROOTROT_END_INDEX + (BODY_JOINTS - 1) * 6
BODYJOINT_END_INDEX = BODYROT_END_INDEX + BODY_JOINTS * 3
BODYJOINTVEL_END_INDEX = BODYJOINT_END_INDEX + (BODY_JOINTS-1) * 3

BODYHANDROT_END_INDEX = ROOTROT_END_INDEX + (BODYHAND_JOINTS - 1) * 6
BODYHANDJOINT_END_INDEX = BODYHANDROT_END_INDEX + BODYHAND_JOINTS * 3
BODYHANDJOINTVEL_END_INDEX = BODYHANDJOINT_END_INDEX + (BODYHAND_JOINTS - 1) * 3
FOOT_CONTACT_DIM = 4
BODYHANDJOINT2BODYJOINT_END_INDEX = BODYHANDROT_END_INDEX + BODY_JOINTS * 3
BODYHANDJOINTVEL2BODYJOINTVEL_END_INDEX = BODYHANDJOINT_END_INDEX + (BODY_JOINTS - 1) * 3
MOTION_FEATS_BODY_ONLY = 268
MOTION_FEATS_BODYHAND = 664
MOTION_INPUT_FEATS_HANDS_TRAINING = BODYROT_END_INDEX + (BODY_JOINTS * 2)
MOTION_OUTPUT_FEATS_HANDS_TRAINING = HAND_JOINTS * 2 * 6

REYE_END_INDEX = BODYROT_END_INDEX + 18
LEFT_HAND_ROT_END_INDEX = REYE_END_INDEX + (HAND_JOINTS * 6)
RIGHT_HAND_ROT_END_INDEX = LEFT_HAND_ROT_END_INDEX + (HAND_JOINTS * 6)

trans_matrix_xyz_to_xzy = to_tensor(torch.Tensor([[1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [0.0, -1.0, 0.0]]), device=device)

trans_matrix_xzy_to_xyz = to_tensor(torch.Tensor([[1.0, 0.0, 0.0],
                                        [0.0, 0.0, -1.0],
                                        [0.0, 1.0, 0.0]]), device=device)

def delta_rel_trans_mot2mot_feats(mot_init, p0_delta_, p1_delta_):
    p0_delta_[:, 0:1, :3] = p0_delta_[:, 0:1, :3] + mot_init[:, :, :3] 
    # motions = torch.cat((mot_init, delta_mots), dim=1)
    p0_ = torch.cat((torch.cumsum(p0_delta_[..., :3], dim=1), p0_delta_[..., 3:]), dim=-1)
    p1_ = p1_delta_
    p1_[:, :, :3] = p1_delta_[:, :, :3] + p0_[:, :, :3]
    return p0_, p1_


def deltafirst_rel_trans_mot2mot_feats(mot_init, p0_delta_, p1_delta_):
    p0_ = p0_delta_
    p0_[:, :, :3] = mot_init[:, :, :3] + p0_delta_[:, :, :3]
    p1_ = p1_delta_
    p1_[:, :, :3] = p1_delta_[:, :, :3] + p0_[:, :, :3]
    return p0_, p1_


def deltatrans_mot2mot_feats(mot_init, delta_mots):
    delta_mots[:, 0:1, :3] = delta_mots[:, 0:1, :3] + mot_init[:, :, :3] 
    # motions = torch.cat((mot_init, delta_mots), dim=1)
    return torch.cat((torch.cumsum(delta_mots[..., :3], dim=1), delta_mots[..., 3:]), dim=-1)

def delta_mot2mot_feats(mot_init, delta_mots):
    delta_mots[:, 0:1, :9] = delta_mots[:, 0:1, :9] + mot_init[:, :, :9] 
    # motions = torch.cat((mot_init, delta_mots), dim=1)
    return torch.cat((torch.cumsum(delta_mots[..., :9], dim=1), delta_mots[..., 9:]), dim=-1)

def mot_to_smplx(motion, betas, gender, with_fingers=False, root_offset=None, interpolate_by=None):
    assert motion.ndim == 2
    T = len(motion)
    if with_fingers:
        motion = motion[..., :BODYHANDROT_END_INDEX]
    else:
        motion = motion[..., :BODYROT_END_INDEX]
    if interpolate_by:
        T1 = T * interpolate_by
        x = np.linspace(0, T-1 ,T)
        x_new = np.linspace(0, T-1 ,T1)
        motion_int = np.zeros((T1, motion.shape[-1]))
        for v1 in range(0, motion.shape[-1]):
            p_ = to_np(motion[:, v1])
            f_p = interpolate.interp1d(x, p_, kind = 'linear')
            motion_int[:, v1] = f_p(x_new)
        motion = to_tensor(motion_int, device=motion.device)
        T = T1
    betas = betas.repeat(T, 1)

    root_transl = motion[:, :ROOTTRANS_END_INDEX]
    if root_offset is not None:
        root_transl = root_transl - root_offset
    root_orient = d62aa(motion[:, ROOTTRANS_END_INDEX:ROOTROT_END_INDEX])
    poses = to_tensor(torch.zeros((T, BODYHAND_JOINTS*3)), device=device)
    poses[:, :3] = root_orient
    if with_fingers:         # only body
        poses[:, 1*3:BODYHAND_JOINTS*3] = d62aa(motion[:, ROOTROT_END_INDEX:BODYHANDROT_END_INDEX].reshape(T, (BODYHAND_JOINTS-1), 6) ).reshape(T, (BODYHAND_JOINTS-1)*3)
    else:    
        poses[:, 1*3:BODY_JOINTS*3] = d62aa(
            motion[:, ROOTROT_END_INDEX:BODYROT_END_INDEX].reshape(T, (BODY_JOINTS-1), 6) ).reshape(T, (BODY_JOINTS-1)*3)
    data = {
        'transl': to_np(root_transl),
        'global_orient': to_np(root_orient),
        'poses': to_np(poses),
        'betas': to_np(betas),
        'meta': {'gender': gender}
    }
    return data


def smplx_batchify(data):
    B = data['transl'].shape[0]
    T = data['transl'].shape[1]
    batchified_data = {
        'betas': data['betas'].reshape(B*T, 10),
        'transl': data['transl'].reshape(B*T, 3),
        'global_orient': data['global_orient'].reshape(B*T, 1, 3, 3),
        'body_pose': data['body_pose'].reshape(B*T, BODY_JOINTS - 1, 3, 3),
        'jaw_pose' : data['jaw_pose'].reshape(B*T, 1, 3, 3),
        'leye_pose' : data['leye_pose'].reshape(B*T, 1, 3, 3),
        'reye_pose' : data['reye_pose'].reshape(B*T, 1, 3, 3),
        'left_hand_pose' : data['left_hand_pose'].reshape(B*T, HAND_JOINTS, 3, 3),
        'right_hand_pose' : data['right_hand_pose'].reshape(B*T, HAND_JOINTS, 3, 3),

    }
    return batchified_data
    
def batch_mot_to_smplx(motion, betas, with_fingers=False, device='cpu'):
    B, T, dim = motion.shape
    root_transl = to_tensor(motion[..., :ROOTTRANS_END_INDEX], device=device)
    root_orient = to_tensor(d62rotmat(motion[..., ROOTTRANS_END_INDEX:ROOTROT_END_INDEX]).reshape(B, T, 3, 3), device=device)
    poses = to_tensor(torch.zeros((B, T, BODYHAND_JOINTS, 3, 3)), device=device)
    poses[:, :, 0] = root_orient
    if with_fingers:         # only body
        poses[:, :, 1:BODYHAND_JOINTS] = d62rotmat(motion[..., ROOTROT_END_INDEX:BODYHANDROT_END_INDEX].reshape(B, T, BODYHAND_JOINTS - 1, 6) ).reshape(B, T, BODYHAND_JOINTS - 1, 3, 3)
    else:
        poses[:, :, 1:BODY_JOINTS] = d62rotmat(motion[..., ROOTROT_END_INDEX:BODYROT_END_INDEX].reshape(B, T, BODY_JOINTS - 1, 6) ).reshape(B, T, BODY_JOINTS - 1, 3, 3)
    betas = betas.repeat(1, T, 1)
    data = {
        'betas': betas.reshape(B, T, 10),
        'transl': root_transl.reshape(B, T, 3),
        'global_orient': root_orient.reshape(B, T, 1, 3, 3),
        'body_pose': poses[:, :, 1:BODY_JOINTS].reshape(B, T, BODY_JOINTS - 1, 3, 3),
        'jaw_pose' : poses[:, :, BODY_JOINTS:23].reshape(B, T, 1, 3, 3),
        'leye_pose' : poses[:, :, 23:24].reshape(B, T, 1, 3, 3),
        'reye_pose' : poses[:, :, 24:25].reshape(B, T, 1, 3, 3),
        'left_hand_pose' : poses[:, :, 25:40].reshape(B, T, HAND_JOINTS, 3, 3),
        'right_hand_pose' : poses[:, :, 40:BODYHAND_JOINTS].reshape(B, T, HAND_JOINTS, 3, 3),

    }
    return data


def batch_smplx_to_pos3d(data, smplx_model):
    
    keypoints3d = smplx_model.forward(
        global_orient=to_tensor(data['global_orient'].reshape(-1, 3), device=device),
        body_pose=to_tensor(data['poses'][..., 3:66].reshape(-1, 63), device=device),
        jaw_pose=to_tensor(data['poses'][..., 66:69].reshape(-1, 3), device=device),
        leye_pose=to_tensor(data['poses'][..., 69:72].reshape(-1, 3), device=device),
        reye_pose=to_tensor(data['poses'][..., 72:75].reshape(-1, 3), device=device),
        left_hand_pose=to_tensor(data['poses'][..., 75:120].reshape(-1, 45), device=device),
        right_hand_pose=to_tensor(data['poses'][..., 120:].reshape(-1, 45), device=device),
        transl=to_tensor(data['transl'].reshape(-1, 3), device=device),
        betas=to_tensor(data['betas'][..., :10].reshape(-1, 10), device=device)
        ).joints
    nframes = keypoints3d.shape[0]
    return keypoints3d

def smplx_to_pos3d(data):
    
    smplx = SMPLX(model_path='models/smplx', betas=data['betas'][:, :10], gender=data['meta']['gender'], \
        batch_size=len(data['betas']), num_betas=10, use_pca=False, use_face_contour=True, flat_hand_mean=True).to(device)

    keypoints3d = smplx.forward(
        global_orient=to_tensor(data['global_orient'], device=device),
        body_pose=to_tensor(data['poses'][:, 3:66], device=device),
        jaw_pose=to_tensor(data['poses'][:, 66:69], device=device),
        leye_pose=to_tensor(data['poses'][:, 69:72], device=device),
        reye_pose=to_tensor(data['poses'][:, 72:75], device=device),
        left_hand_pose=to_tensor(data['poses'][:, 75:120], device=device),
        right_hand_pose=to_tensor(data['poses'][:, 120:], device=device),
        transl=to_tensor(data['transl'], device=device),
        betas=to_tensor(data['betas'][:, :10], device=device)
        ).joints[:, :BODYHAND_JOINTS]
    nframes = keypoints3d.shape[0]
    return keypoints3d

def root_invariant_smplx_to_pos3d(data):
    smplx = SMPLX(model_path='models/smplx', betas=data['betas'][:, :10], gender=data['meta']['gender'], \
        batch_size=len(data['betas']), num_betas=10, use_pca=False, use_face_contour=True, flat_hand_mean=True).to(device)
    keypoints3d = smplx.forward(
        body_pose=to_tensor(data['poses'][:, 3:66], device=device),
        jaw_pose=to_tensor(data['poses'][:, 66:69], device=device),
        leye_pose=to_tensor(data['poses'][:, 69:72], device=device),
        reye_pose=to_tensor(data['poses'][:, 72:75], device=device),
        left_hand_pose=to_tensor(data['poses'][:, 75:120], device=device),
        right_hand_pose=to_tensor(data['poses'][:, 120:], device=device),
        betas=to_tensor(data['betas'][:, :10], device=device)
        ).joints[:, :BODYHAND_JOINTS]
    nframes = keypoints3d.shape[0]
    return keypoints3d


def get_smplx_default_joint_offset(data):
    smplx = SMPLX(model_path='models/smplx', 
                #   betas=data['betas'][:, :10],
                  gender=data['meta']['gender'], 
                  batch_size=len(data['betas']), 
                  num_betas=10, 
                  use_pca=False, 
                  use_face_contour=True, 
                  flat_hand_mean=True).to(device)

    keypoints3d = smplx.forward(
        global_orient=to_tensor(data['global_orient'], device=device),
        body_pose=to_tensor(data['poses'][:, 3:66], device=device),
        jaw_pose=to_tensor(data['poses'][:, 66:69], device=device),
        leye_pose=to_tensor(data['poses'][:, 69:72], device=device),
        reye_pose=to_tensor(data['poses'][:, 72:75], device=device),
        left_hand_pose=to_tensor(data['poses'][:, 75:120], device=device),
        right_hand_pose=to_tensor(data['poses'][:, 120:], device=device),
        betas=to_tensor(data['betas'][:, :10], device=device)
        ).joints[:, :BODYHAND_JOINTS]
    
    return keypoints3d


def normalize_vec(x):
    if not torch.is_tensor(x):
        x = to_tensor(x, device=device)
    return x / torch.linalg.norm(x)


def align_mot_z(data, n_joints=BODYHAND_JOINTS):
    rootjoint_offset = get_smplx_default_joint_offset(data)[0, 0]
    data_rotated = data
    joint_pos_ = smplx_to_pos3d(data)
    joint_pos_interchnge_axis = torch.einsum("mn, tjn->tjm", trans_matrix_xyz_to_xzy, joint_pos_) #interchanging Y and Z axis
    root_rot_change_axis = torch.einsum("mn, tn->tm", trans_matrix_xyz_to_xzy, data['global_orient'])
    
    #Sanity check
    myquat_b = axis_angle_to_quaternion(data['poses'].reshape(-1, BODYHAND_JOINTS , 3))
    root_quat_change_axis = axis_angle_to_quaternion(root_rot_change_axis)
    # before_change_axis = qrot(myquat_b[0, 0], joint_pos_[0,0])
    # aftr_change_axis = qrot(root_quat_change_axis[0], joint_pos_interchnge_axis[0,0])
    
    # # # '''Put on Floor'''
    # floor_height = joint_pos_interchnge_axis.min(axis=0).min(axis=0)[1]
    # joint_pos_interchnge_axis[:, :, 1] -= floor_height

    '''XZ at origin'''
    root_pos_init = joint_pos_interchnge_axis[0]
    root_pos_init_xz = root_pos_init[0] * to_tensor(np.array([1, 0, 1]), device=device)
    joint_pos_translated = joint_pos_interchnge_axis - root_pos_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across = root_pos_init[r_hip] - root_pos_init[l_hip]
    across = normalize_vec(across)
    # pelvis, spine = up_vec_idx
    # up = root_pos_init[spine] - root_pos_init[pelvis]
    # up = normalize_vec(up)
    # forward (3,), rotate around y-axis
    forward_init = torch.linalg.cross(to_tensor(np.array([0, 1, 0]), device=device), across, dim=-1)
    # forward_init = np.cross(up, across, axis=-1)
    # forward (3,)
    forward_init = normalize_vec(forward_init)
    target = to_tensor(np.array([0, 0, 1]), device=device)
    root_quat_init = qbetween(forward_init, target)   # angle = np.arccos(forward_init[0, 2]) iff target = [0, 0, 1]
    root_quat_init_for_all = to_tensor(np.ones(joint_pos_interchnge_axis.shape[:-1] + (4,)), device=device) * root_quat_init
    root_quat_init_for_rot = to_tensor(np.ones(root_quat_change_axis.shape[:-1] + (4,)), device=device) * root_quat_init
    joint_pos_rotated = qrot(root_quat_init_for_all, joint_pos_translated)
    root_rot_multiply_quat_init = qmul(root_quat_init_for_rot, root_quat_change_axis)
    
    root_rot_change_back_axis = torch.einsum("mn, tn->tm", trans_matrix_xzy_to_xyz, quaternion_to_axis_angle(root_rot_multiply_quat_init))
    root_trans_change_back_axis = torch.einsum("mn, tn->tm", trans_matrix_xzy_to_xyz, joint_pos_rotated[:, 0])
    
    data_rotated['transl'] = root_trans_change_back_axis - rootjoint_offset
    data_rotated['global_orient'] = root_rot_change_back_axis
    data_rotated['poses'][:, :3] = root_rot_change_back_axis

    global_joint_positions = joint_pos_rotated.reshape(len(joint_pos_rotated), -1)
    # data_rotated['transl'] = joint_pos_rotated[:, 0] - np.einsum("mn, tn->tm", trans_matrix_xyz_to_xzy, rootjoint_offset[None])
    # data_rotated['global_orient'] = to_np(quaternion_to_axis_angle(root_rot_multiply_quat_init))
    # data_rotated['poses'][:, :3] = to_np(quaternion_to_axis_angle(root_rot_multiply_quat_init))
    
    return data_rotated, global_joint_positions, root_quat_init, root_pos_init_xz, rootjoint_offset


def rigid_transform(relative, data, motion, n_joints=BODYHAND_JOINTS ):
    rootjoint_offset = get_smplx_default_joint_offset(data)[0, 0]
    data_transformed = data

    root_rot_change_axis = torch.einsum("mn, tn->tm", trans_matrix_xyz_to_xzy, data['global_orient'])
    root_quat_change_axis = axis_angle_to_quaternion(root_rot_change_axis)
    
    global_positions = motion[..., :n_joints * 3].reshape(motion.shape[:-1] + (n_joints, 3))

    relative_rot = relative[0]
    relative_rot_cos = torch.cos(relative_rot)
    relative_rot_sin = torch.sin(relative_rot)
    relative_t = relative[1:3]

    relative_r_rot_quat_for_positions = to_tensor(np.zeros(global_positions.shape[:-1] + (4,)), device=device)
    relative_r_rot_quat_for_positions[..., 0] = relative_rot_cos
    relative_r_rot_quat_for_positions[..., 2] = relative_rot_sin
    inv_relative_r_rot_quat_for_positions = qinv(relative_r_rot_quat_for_positions)
    global_positions_relative_rotated = qrot(inv_relative_r_rot_quat_for_positions, global_positions)
    global_positions_relative_translated = global_positions_relative_rotated.clone()
    global_positions_relative_translated[..., [0, 2]] = global_positions_relative_rotated[..., [0, 2]] + relative_t
    
    relative_r_rot_quat_for_rotation = to_tensor(np.zeros(root_quat_change_axis.shape[:-1] + (4,)), device=device)
    relative_r_rot_quat_for_rotation[..., 0] = relative_rot_cos
    relative_r_rot_quat_for_rotation[..., 2] = relative_rot_sin
    inv_relative_r_rot_quat_for_rotation = qinv(relative_r_rot_quat_for_rotation)
    root_quat_relative_rotated = qmul(inv_relative_r_rot_quat_for_rotation, root_quat_change_axis)


    root_rot_change_back_axis = torch.einsum("mn, tn->tm", trans_matrix_xzy_to_xyz,
                                           quaternion_to_axis_angle(root_quat_relative_rotated))
    root_trans_change_back_axis = torch.einsum("mn, tn->tm", trans_matrix_xzy_to_xyz, global_positions_relative_translated[:, 0])
    
    data_transformed['transl'] = root_trans_change_back_axis - rootjoint_offset
    data_transformed['global_orient'] = root_rot_change_back_axis
    data_transformed['poses'][:, :3] = root_rot_change_back_axis    
    return data_transformed, global_positions_relative_translated

def reduced_process_mot(data, with_fingers=True):
    root_transl = data[..., :ROOTTRANS_END_INDEX]
    root_rot = data[..., ROOTTRANS_END_INDEX:ROOTROT_END_INDEX]
    if with_fingers:
        body_rot = data[..., ROOTROT_END_INDEX:BODYHANDROT_END_INDEX]
    else:
        body_rot = data[..., ROOTROT_END_INDEX:BODYROT_END_INDEX]
    return root_transl, root_rot, body_rot


def reverse_process_mot(data, with_fingers=True):
    root_transl = data[..., :ROOTTRANS_END_INDEX]
    root_rot = data[..., ROOTTRANS_END_INDEX:ROOTROT_END_INDEX]
    foot_contact = data[..., -FOOT_CONTACT_DIM:]
    if with_fingers:
        body_rot = data[..., ROOTROT_END_INDEX:BODYHANDROT_END_INDEX]
        local_jointpos = data[..., BODYHANDROT_END_INDEX:BODYHANDJOINT_END_INDEX]
        local_jointvel = data[..., BODYHANDJOINT_END_INDEX:BODYHANDJOINTVEL_END_INDEX]
    else:
        body_rot = data[..., ROOTROT_END_INDEX:BODYROT_END_INDEX]
        local_jointpos = data[..., BODYROT_END_INDEX:BODYJOINT_END_INDEX]
        local_jointvel = data[..., BODYJOINT_END_INDEX:BODYJOINTVEL_END_INDEX]
    return root_transl, root_rot, body_rot, local_jointpos, local_jointvel, foot_contact


# def process_mot(data, feet_thre=0.001, with_fingers=True):
#     trans_joint_offsets = get_smplx_default_joint_offset(data)[:, 0]
#     joint_pos_ = smplx_to_pos3d(data)
#     '''Get Binary Foot Contacts ''' 
#     def foot_detect(positions, thres):
#         velfactor, heightfactor = to_tensor(np.array([thres, thres]), device=device), to_tensor(np.array([0.12, 0.05]), device=device)
#         feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
#         feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
#         feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
#         feet_l_h = positions[:-1,fid_l, 2]
#         feet_l = to_tensor(np.zeros((len(positions), 2)), device=device)
#         feet_r = to_tensor(np.zeros((len(positions), 2)), device=device)
#         feet_l[1:] = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor))
#         feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
#         feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
#         feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
#         feet_r_h = positions[:-1,fid_r, 2]
#         feet_r[1:] = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor))
#         return feet_l, feet_r
#     feet_l, feet_r = foot_detect(joint_pos_, feet_thre)

#     '''Get Root local joint pose and global joint velocities'''
#     global_joint_pos = joint_pos_ - trans_joint_offsets[:, None]
#     root_local_joint_pos = global_joint_pos - global_joint_pos[:, 0:1]      # for BODYHAND_JOINTS - 1 joints
    
#     joint_vels = joint_pos_.clone()
#     joint_vels[1:] = joint_pos_[1:] - joint_pos_[:-1]
#     '''Get 6D rotation for all joints'''
#     rot6d = aa2d6(data['poses'].reshape(-1, BODYHAND_JOINTS , 3)).to(device)

#     processed_data = global_joint_pos[:, 0]                                                               # Global Root translation, shape=(:, 3)
#     if with_fingers:
#         processed_data = torch.cat([processed_data, rot6d.reshape(len(rot6d), BODYHAND_JOINTS *6)], dim=-1)                              # 6D rotations of all joints, shape=(:, 330)
#         processed_data = torch.cat([processed_data, root_local_joint_pos[:, 1:].reshape(len(joint_pos_), (BODYHAND_JOINTS - 1)*3)], dim=-1)   # Local joint positions wrt root, shape =(:, 162)
#         processed_data = torch.cat([processed_data, joint_vels.reshape(len(joint_vels), BODYHAND_JOINTS *3)], dim=-1)                    # Global joint velocities, shape=(:, 165)
#     else:
#         processed_data = torch.cat([processed_data, rot6d[:, :BODY_JOINTS].reshape(len(rot6d), BODY_JOINTS*6)], dim=-1)                       # 6D rotations of body joints, shape=(:, 132)
#         processed_data = torch.cat([processed_data, root_local_joint_pos[:, 1:BODY_JOINTS].reshape(len(joint_pos_), (BODY_JOINTS - 1)*3)], dim=-1)  # Local joint positions wrt root, shape =(:, 63)
#         processed_data = torch.cat([processed_data, joint_vels[:, :BODY_JOINTS].reshape(len(joint_vels), BODY_JOINTS*3)], dim=-1)             # Global joint velocities, shape=(:, 66)
    
#     processed_data = torch.cat([processed_data, feet_l, feet_r], dim=-1)                           # Binary foot contact, shape=(:, 4)
#     return processed_data

def new_process_mot(data, feet_thre=0.001, with_fingers=True):
    joint_pos_ = root_invariant_smplx_to_pos3d(data)
    '''Get Binary Foot Contacts ''' 
    def foot_detect(positions, thres):
        velfactor, heightfactor = to_tensor(np.array([thres, thres]), device=device), to_tensor(np.array([0.12, 0.05]), device=device)
        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1,fid_l, 2]
        feet_l = to_tensor(torch.zeros((len(positions), 2)), device=device)
        feet_r = to_tensor(torch.zeros((len(positions), 2)), device=device)
        feet_l[1:] = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor))
        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1,fid_r, 2]
        feet_r[1:] = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor))
        return feet_l, feet_r
    feet_l, feet_r = foot_detect(joint_pos_, feet_thre)

    '''Get root invariant local joint velocities'''
    joint_vels = to_tensor(torch.zeros_like(joint_pos_), device=joint_pos_.device)
    joint_vels[1:] = joint_pos_[1:] - joint_pos_[:-1]
    '''Get 6D rotation for all joints'''
    rot6d = aa2d6(data['poses'].reshape(-1, BODYHAND_JOINTS , 3)).to(device)

    processed_data = data['transl']                                                               # Global Root translation, shape=(:, 3)
    if with_fingers:
        processed_data = torch.cat([processed_data, rot6d.reshape(len(rot6d), BODYHAND_JOINTS *6)], dim=-1)                              # 6D rotations of all joints, shape=(:, 330)
        processed_data = torch.cat([processed_data, joint_pos_.reshape(len(joint_pos_), BODYHAND_JOINTS*3)], dim=-1)   # Local joint positions wrt root, shape =(:, 165)
        processed_data = torch.cat([processed_data, joint_vels[:, 1:].reshape(len(joint_vels), (BODYHAND_JOINTS-1)*3)], dim=-1)                    # Global joint velocities, shape=(:, 165)
    else:
        processed_data = torch.cat([processed_data, rot6d[:, :BODY_JOINTS].reshape(len(rot6d), BODY_JOINTS*6)], dim=-1)                       # 6D rotations of body joints, shape=(:, 132)
        processed_data = torch.cat([processed_data, joint_pos_[:, :BODY_JOINTS].reshape(len(joint_pos_), BODY_JOINTS*3)], dim=-1)  # Local joint positions wrt root, shape =(:, 66)
        processed_data = torch.cat([processed_data, joint_vels[:, 1:BODY_JOINTS].reshape(len(joint_vels), (BODY_JOINTS-1)*3)], dim=-1)             # Global joint velocities, shape=(:, 66)
    
    processed_data = torch.cat([processed_data, feet_l, feet_r], dim=-1)                           # Binary foot contact, shape=(:, 4)
    return processed_data

def batchify_foot_contact(positions, thres):
        velfactor, heightfactor = to_tensor(torch.asarray([thres, thres]), device=positions.device), to_tensor(torch.asarray([0.12, 0.05]), device=positions.device)
        feet_l_x = (positions[:, 1:, fid_l, 0] - positions[:, :-1, fid_l, 0]) ** 2
        feet_l_y = (positions[:, 1:, fid_l, 1] - positions[:, :-1, fid_l, 1]) ** 2
        feet_l_z = (positions[:, 1:, fid_l, 2] - positions[:, :-1, fid_l, 2]) ** 2
        feet_l_h = positions[:, :-1,fid_l, 2]
        feet_l = to_tensor(torch.zeros((positions.shape[0], positions.shape[1], 2)), device=positions.device)
        feet_r = to_tensor(torch.zeros((positions.shape[0], positions.shape[1], 2)), device=positions.device)
        feet_l[:, 1:] = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor))
        feet_r_x = (positions[:, 1:, fid_r, 0] - positions[:, :-1, fid_r, 0]) ** 2
        feet_r_y = (positions[:, 1:, fid_r, 1] - positions[:, :-1, fid_r, 1]) ** 2
        feet_r_z = (positions[:, 1:, fid_r, 2] - positions[:, :-1, fid_r, 2]) ** 2
        feet_r_h = positions[:, :-1,fid_r, 2]
        feet_r[:, 1:] = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor))
        return torch.cat([feet_l, feet_r], dim=-1)

def smooth_root_joint(sequence, root_joint_idx=0, window_size=5):
    """
    Apply a moving average filter to the root joint in a human motion sequence.

    Args:
        sequence (torch.Tensor): The motion sequence of shape [B, T, 3*J].
        root_joint_idx (int): Index of the root joint (default is 0).
        window_size (int): Size of the moving average window (default is 5).

    Returns:
        torch.Tensor: Smoothed motion sequence.
    """
    # Extract root joint positions (assuming 3D)
    root_joint_positions = sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3]
    
    # Create a kernel for each of the three dimensions
    kernel = torch.ones(3, 1, window_size) / window_size  # Shape: [3, 1, window_size]
    kernel = kernel.to(sequence.device)
    # Apply 1D convolution along the time dimension for each batch and each root joint coordinate
    smoothed_positions = F.conv1d(
        root_joint_positions.permute(0, 2, 1),  # Reshape to [B, 3, T]
        kernel, padding=window_size // 2, groups=3
    ).permute(0, 2, 1)  # Reshape back to [B, T, 3]

    # Replace the root joint positions in the original sequence with smoothed values
    smoothed_sequence = sequence.clone()
    smoothed_sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3] = smoothed_positions

    return smoothed_sequence


import torch

def batch_smooth_root_joint_constrained(sequence, root_joint_idx=0, window_size=5, anchor_interval=10):
    """
    Apply a constrained moving average filter to the root joint in a human motion sequence.
    Anchors the root joint at specified intervals to prevent drift.

    Args:
        sequence (torch.Tensor): The motion sequence of shape [B, T, 3*J].
        root_joint_idx (int): Index of the root joint (default is 0).
        window_size (int): Size of the moving average window (default is 5).
        anchor_interval (int): Interval for anchor points (default is 10).

    Returns:
        torch.Tensor: Smoothed motion sequence.
    """
    # Extract root joint positions (assuming 3D)
    root_joint_positions = sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3]  # Shape: [B, T, 3]
    
    # Initialize smoothed positions with the original
    smoothed_positions = root_joint_positions.clone()
    
    # Iterate over frames, applying a moving average while anchoring
    for start in range(0, root_joint_positions.size(1) - window_size, anchor_interval):
        end = min(start + window_size, root_joint_positions.size(1))
        
        # Apply smoothing only between anchor points
        segment = root_joint_positions[:, start:end]
        smoothed_segment = torch.stack([
            segment[:, :, i].unfold(1, window_size, 1).mean(dim=2)
            for i in range(3)
        ], dim=2)  # Shape: [B, end-start, 3]
        
        # Update the smoothed segment in the output
        smoothed_positions[:, start:end] = smoothed_segment
    
    # Blend original positions at anchor points for constraint
    for start in range(0, root_joint_positions.size(1), anchor_interval):
        smoothed_positions[:, start] = root_joint_positions[:, start]
    
    # Update the original sequence with the smoothed root joint positions
    smoothed_sequence = sequence.clone()
    smoothed_sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3] = smoothed_positions

    return smoothed_sequence

def smooth_root_joint_constrained(root_joint_positions, window_size=5, anchor_interval=60):
    """
    Apply a constrained moving average filter to the root joint in a human motion sequence.
    Anchors the root joint at specified intervals to prevent drift.

    Args:
        root_joint_positions (torch.Tensor): The motion sequence of shape [B, T, 3].
        
        window_size (int): Size of the moving average window (default is 5).
        anchor_interval (int): Interval for anchor points (default is 10).

    Returns:
        torch.Tensor: Smoothed motion sequence.
    """
    
    # Initialize smoothed positions with the original
    smoothed_positions = root_joint_positions.clone()
    
    # Iterate over frames, applying a moving average while anchoring
    for start in range(0, root_joint_positions.size(1) - window_size, anchor_interval):
        end = min(start + window_size, root_joint_positions.size(1))
        
        # Apply smoothing only between anchor points
        segment = root_joint_positions[:, start:end]
        smoothed_segment = torch.stack([
            segment[:, :, i].unfold(1, window_size, 1).mean(dim=2)
            for i in range(3)
        ], dim=2)  # Shape: [B, end-start, 3]
        
        # Update the smoothed segment in the output
        smoothed_positions[:, start:end] = smoothed_segment
    
    # Blend original positions at anchor points for constraint
    for start in range(0, root_joint_positions.size(1), anchor_interval):
        smoothed_positions[:, start] = root_joint_positions[:, start]

    return smoothed_positions


def smooth_root_joint_deltas(sequence, root_joint_idx=0, alpha_xz=0.5, alpha_y=0.1):
    """
    Apply a smoothing filter to the root joint's X, Y, and Z deltas, preserving the initial position.
    This reduces jumps by smoothing the changes in position instead of the absolute positions.

    Args:
        sequence (torch.Tensor): The motion sequence of shape [B, T, 3*J].
        root_joint_idx (int): Index of the root joint (default is 0).
        alpha_xz (float): Smoothing factor for X and Z deltas (higher = more smoothing).
        alpha_y (float): Smoothing factor for Y deltas (higher = more smoothing).

    Returns:
        torch.Tensor: Smoothed motion sequence.
    """
    # Extract root joint positions (assuming 3D)
    root_joint_positions = sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3]  # Shape: [B, T, 3]

    # Calculate deltas (differences) between consecutive frames
    deltas = root_joint_positions[:, 1:] - root_joint_positions[:, :-1]  # Shape: [B, T-1, 3]

    # Separate XZ and Y deltas
    xz_deltas = deltas[:, :, [0, 2]]
    y_deltas = deltas[:, :, 1:2]

    # Initialize smoothed deltas with the original deltas
    smoothed_xz_deltas = xz_deltas.clone()
    smoothed_y_deltas = y_deltas.clone()

    # Apply exponential smoothing on the deltas
    for t in range(1, deltas.size(1)):
        smoothed_xz_deltas[:, t] = alpha_xz * xz_deltas[:, t] + (1 - alpha_xz) * smoothed_xz_deltas[:, t - 1]
        smoothed_y_deltas[:, t] = alpha_y * y_deltas[:, t] + (1 - alpha_y) * smoothed_y_deltas[:, t - 1]

    # Reconstruct positions by cumulatively summing the smoothed deltas, starting from the initial position
    smoothed_positions = torch.cat([root_joint_positions[:, :1],  # Initial position
                                    root_joint_positions[:, :1] + torch.cumsum(
                                        torch.cat([smoothed_y_deltas, smoothed_xz_deltas], dim=2), dim=1)], dim=1)

    # Insert the smoothed root joint positions back into the original sequence
    smoothed_sequence = sequence.clone()
    smoothed_sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3] = smoothed_positions

    return smoothed_sequence


def smooth_root_joint_XZ(sequence, root_joint_idx=0, window_size=5, y_window_size=3):
    """
    Apply a moving average filter to the root joint's X and Z directions in a batched human motion sequence,
    with minimal smoothing on Y to avoid jumps, ensuring the initial frames stay consistent.

    Args:
        sequence (torch.Tensor): The motion sequence of shape [B, T, 3*J].
        root_joint_idx (int): Index of the root joint (default is 0).
        window_size (int): Size of the moving average window for X and Z directions.
        y_window_size (int): Size of the moving average window for Y direction (default is 3).

    Returns:
        torch.Tensor: Smoothed motion sequence with frame 0 unchanged for the root joint.
    """
    # Extract root joint positions (assuming 3D)
    root_joint_positions = sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3]  # Shape: [B, T, 3]
    
    # Keep the original root position at frame 0
    initial_position = root_joint_positions[:, :1, :]  # Shape: [B, 1, 3]
    
    # Create kernels for the X and Z directions
    kernel_xz = torch.ones(2, 1, window_size) / window_size  # Shape: [2, 1, window_size]
    kernel_xz = kernel_xz.to(sequence.device)

    # Create a kernel for the Y direction with minimal smoothing
    kernel_y = torch.ones(1, 1, y_window_size) / y_window_size  # Shape: [1, 1, y_window_size]
    kernel_y = kernel_y.to(sequence.device)
    
    # Extract X, Y, and Z positions separately, excluding frames 0 and 1
    xz_positions = root_joint_positions[:, 2:, [0, 2]]  # Shape: [B, T-2, 2]
    y_position = root_joint_positions[:, 2:, 1:2]       # Shape: [B, T-2, 1]

    # Apply 1D convolution to smooth the X and Z components from frame 2 onward
    padding_xz = (window_size - 1) // 2
    smoothed_xz = F.conv1d(
        xz_positions.permute(0, 2, 1),  # Reshape to [B, 2, T-2]
        kernel_xz, padding=padding_xz, groups=2
    ).permute(0, 2, 1)  # Reshape back to [B, T-2, 2]

    # Apply minimal smoothing on the Y component
    padding_y = (y_window_size - 1) // 2
    smoothed_y = F.conv1d(
        y_position.permute(0, 2, 1),  # Reshape to [B, 1, T-2]
        kernel_y, padding=padding_y
    ).permute(0, 2, 1)  # Reshape back to [B, T-2, 1]

    # Concatenate the initial position at frame 0 and keep frame 1 the same, then smooth from frame 2 onward
    smoothed_positions = root_joint_positions.clone()
    smoothed_positions[:, 2:, [0, 2]] = smoothed_xz
    smoothed_positions[:, 2:, 1:2] = smoothed_y
    smoothed_positions[:, 1:2, 1:2] = initial_position[:, :, 1:2]  # Keep frame 1's Y the same as frame 0's Y

    # Replace the root joint positions in the original sequence with smoothed values
    smoothed_sequence = sequence.clone()
    smoothed_sequence[:, :, root_joint_idx*3:(root_joint_idx+1)*3] = smoothed_positions

    return smoothed_sequence

def prevent_foot_sliding_root_positions(motion_sequence, foot_indices, threshold=0.1):
    """
    Adjusts the root positions in a human motion sequence to prevent foot sliding based on vertical movement.

    Args:
        motion_sequence (torch.Tensor): A tensor of shape [T, J, 3] representing the motion sequence.
        foot_indices (list): Indices of the joints representing the feet (e.g., [1, 2] for left and right feet).
        threshold (float): Maximum allowable movement of the root joint when feet are stationary in the vertical direction.

    Returns:
        torch.Tensor: Adjusted root positions of shape [T, 3].
    """
    # Extract the number of frames
    T, J, _ = motion_sequence.shape
    
    # Extract root positions
    root_positions = motion_sequence[:, 0, :].clone()
    
    # Loop through the frames and adjust the root position if needed
    for t in range(1, T):
        # Calculate the vertical movement of the foot joints
        foot_positions_current = motion_sequence[t, foot_indices, 1]  # Y-coordinates of current frame
        foot_positions_previous = motion_sequence[t - 1, foot_indices, 1]  # Y-coordinates of previous frame
        foot_movement = foot_positions_current - foot_positions_previous
        
        # Check if vertical foot movement is below the threshold
        if torch.all(torch.abs(foot_movement) < threshold):
            root_positions[t] = root_positions[t - 1]  # Lock root position

    return root_positions

def find_best_partial_match(lookup_table, batch_code0):
    indices = []
    code_idx_0 = batch_code0.flatten()
    
    for idx in code_idx_0:
        best_idx = -1  # Default for no match
        for i, (lookup_pair_0, lookup_pair_1) in enumerate(lookup_table):
            if idx == lookup_pair_0:  # Check for a match on the first element
                best_idx = i
                break  # Stop at the first best match
        indices.append(best_idx)
    
    return indices

def search_lookuptable_partial(batch_code0, lookup_table):
    bs = batch_code0.shape[0]
    token_indices = find_best_partial_match(lookup_table, batch_code0)
    return to_tensor(token_indices, device=batch_code0.device).int().view(bs, -1)