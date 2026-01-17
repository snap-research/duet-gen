import torch
import torch.nn.functional as F
import math
from einops import rearrange

# return mask where padding is FALSE
def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask #(b, len)

# return mask where padding is ALL FALSE
def get_pad_mask_idx(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1)

# Given seq: (b, s)
# Return mat: (1, s, s)
# Example Output:
#        [[[ True, False, False],
#          [ True,  True, False],
#          [ True,  True,  True]]]
# For causal attention
def get_subsequent_mask(seq):
    sz_b, seq_len = seq.shape
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
    return subsequent_mask.to(seq.device)


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def l2norm(t):
    return F.normalize(t, dim = -1)

# tensor helpers

# Get a random subset of TRUE mask, with prob
def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask


# Get mask of special_tokens in ids
def get_mask_special_tokens(ids, special_ids):
    mask = torch.zeros_like(ids).bool()
    for special_id in special_ids:
        mask |= (ids==special_id)
    return mask

# network builder helpers
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

# classifier free guidance functions

def uniform(shape, device=None, r1=0, r2=1):
    return torch.zeros(shape, device=device).float().uniform_(r1, r2)

def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = 1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


# Example input:
#        [[ 0.3596,  0.0862,  0.9771, -1.0000, -1.0000, -1.0000],
#         [ 0.4141,  0.1781,  0.6628,  0.5721, -1.0000, -1.0000],
#         [ 0.9428,  0.3586,  0.1659,  0.8172,  0.9273, -1.0000]]
# Example output:
#        [[  -inf,   -inf, 0.9771,   -inf,   -inf,   -inf],
#         [  -inf,   -inf, 0.6628,   -inf,   -inf,   -inf],
#         [0.9428,   -inf,   -inf,   -inf,   -inf,   -inf]]
def top_k(logits, thres = 0.9, dim = 1):
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim = dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    # func verified
    # print(probs)
    # print(logits)
    # raise
    return probs

# noise schedules

# More on large value, less on small
def linear_schedule(t):
    return t

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def scale_cosine_schedule(t, scale):
    return torch.clip(scale*torch.cos(t * math.pi * 0.5) + 1 - scale, min=0., max=1.)

# More on small value, less on large
def q_schedule(bs, low, high, device):
    noise = uniform((bs,), device=device)
    schedule = 1 - cosine_schedule(noise)
    return torch.round(schedule * (high - low - 1)).long() + low

def calc_performance(pred, labels, ignore_index=None, smoothing=0., tk=1):
    loss = cal_loss(pred, labels, ignore_index, smoothing=smoothing)
    pred_id_k = torch.topk(pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]
    mask = labels.ne(ignore_index)
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()
    return loss, pred_id, acc


def cal_loss(pred, labels, ignore_index=None, smoothing=0.):
    '''Calculate cross entropy loss, apply label smoothing if needed.'''
    # print(pred.shape, labels.shape) #torch.Size([64, 1028, 55]) torch.Size([64, 55])
    # print(pred.shape, labels.shape) #torch.Size([64, 1027, 55]) torch.Size([64, 55])
    if smoothing:
        space = 2
        n_class = pred.size(1)
        mask = labels.ne(ignore_index)
        one_hot = rearrange(F.one_hot(labels, n_class + space), 'a ... b -> a b ...')[:, :n_class]
        # one_hot = torch.zeros_like(pred).scatter(1, labels.unsqueeze(1), 1)
        sm_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        neg_log_prb = -F.log_softmax(pred, dim=1)
        loss = (sm_one_hot * neg_log_prb).sum(dim=1)
        # loss = F.cross_entropy(pred, sm_one_hot, reduction='none')
        loss = torch.mean(loss.masked_select(mask))
    else:
        loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)

    return loss


class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weights=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights

    def forward(self, logits, targets, ignore_index):
        # Calculate standard cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, weight=self.weights, ignore_index=ignore_index)
        
        # Calculate the proximity loss
        # Assuming targets are class indices
        probs = F.softmax(logits, dim=1)
        one_hot_targets = F.one_hot(targets, num_classes=logits.size(1)).float()
        proximity_loss = F.mse_loss(probs, one_hot_targets)

        # Combine the losses (adjust the weighting as needed)
        total_loss = ce_loss + proximity_loss
        return total_loss

import torch
import torch.nn.functional as F


def weightedSequenceCrossEntropyLoss(logits, targets, weights, window_size):
    # logits shape: [batch_size, num_classes, T]
    # targets shape: [batch_size, T]

    # Calculate cross-entropy loss for each time step
    ce_loss = F.cross_entropy(logits, targets, weight=weights, reduction='none')
    
    # Initialize proximity loss
    batch_size, num_classes, T = logits.size()
    probs = F.softmax(logits, dim=1)  # Shape: [batch_size, num_classes, T]
    probs = probs.permute(0, 2, 1)
    # Create one-hot encoded targets
    one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()  # Shape: [batch_size, T, num_classes]

    # Calculate proximity loss using a window of size W
    W_half = window_size // 2
    proximity_loss = torch.zeros_like(ce_loss)

    for t in range(W_half, T - W_half):
        # Collect the probabilities within the window
        window_probs = probs[:, t - W_half : t + W_half + 1]  # Shape: [batch_size, W, num_classes]

        # Calculate mean squared error for proximity using the current frame
        proximity_loss[:, t] = (
            sum(F.mse_loss(window_probs[:, w], one_hot_targets[:, t], reduction='none').mean(dim=1) 
                for w in range(window_size))
        ) / window_size  # Average over the window

    # Handle the edges where the full window cannot be applied
    for t in range(0, W_half):
        proximity_loss[:, t] = F.mse_loss(probs[:, t], one_hot_targets[:, t], reduction='none').mean(dim=1)  # Only current frame

    for t in range(T - W_half, T):
        proximity_loss[:, t] = F.mse_loss(probs[:, t], one_hot_targets[:, t], reduction='none').mean(dim=1)  # Only current frame

    # Combine losses
    total_loss = ce_loss + 5.0 * proximity_loss

    return total_loss.mean()  # Return the overall loss

# # Example usage
# T = 10  # Number of time steps
# batch_size = 32
# num_classes = 1024
# window_size = 5  # Size of the window

# logits = torch.randn(T, batch_size, num_classes)  # Logits for T time steps
# targets = torch.randint(0, num_classes, (T, batch_size))  # Random class indices for targets

# # Optional: weights for class imbalance
# weights = torch.ones(num_classes)  # Modify as needed

# loss_fn = WeightedSequenceCrossEntropyLoss(weights=weights, window_size=window_size)
# loss = loss_fn(logits, targets)

