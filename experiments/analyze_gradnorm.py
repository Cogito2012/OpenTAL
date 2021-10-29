import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from AFSD.common.thumos_dataset import THUMOS_Dataset, get_video_info, \
    load_video_data, detection_collate, get_video_anno
from torch.utils.data import DataLoader
from AFSD.thumos14.BDNet import BDNet
from AFSD.thumos14.multisegment_loss import MultiSegmentLoss
from AFSD.common.config import config
import matplotlib.pyplot as plt


def get_basic_config(config):
    class cfg: pass
    cfg.batch_size = config['training']['batch_size']
    cfg.learning_rate = config['training']['learning_rate']
    cfg.weight_decay = config['training']['weight_decay']
    cfg.max_epoch = config['training']['max_epoch']
    cfg.checkpoint_path = config['training']['checkpoint_path']
    cfg.focal_loss = config['training']['focal_loss']
    cfg.edl_loss = config['training']['edl_loss'] if 'edl_loss' in config['training'] else False
    cfg.edl_config = config['training']['edl_config'] if 'edl_config' in config['training'] else None
    cfg.cls_loss_type = 'edl' if cfg.edl_loss else 'focal' # by default, we use focal loss
    cfg.os_head = config['model']['os_head'] if 'os_head' in config['model'] else False
    num_classes = config['dataset']['num_classes']
    cfg.num_classes = num_classes - 1 if cfg.os_head else num_classes
    cfg.random_seed = config['training']['random_seed']
    cfg.train_state_path = os.path.join(cfg.checkpoint_path, 'training')
    assert os.path.exists(cfg.train_state_path)
    cfg.resume = config['training']['resume']
    cfg.use_edl = config['model']['use_edl'] if 'use_edl' in config['model'] else False
    cfg.clip_length = config['dataset']['training']['clip_length']
    cfg.overlap_thresh = config['training']['piou']
    cfg.output_path = config['testing']['output_path']
    return cfg


def get_path(input_path):
    if os.path.lexists(input_path):
        fullpath = os.path.realpath(input_path) if os.path.islink(input_path) else input_path
        real_name = fullpath.split('/')[-1]
        real_full_path = os.path.join(os.path.dirname(input_path), real_name)
    else:
        raise FileNotFoundError
    return real_full_path


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

GLOBAL_SEED = 1

def worker_init_fn(worker_id):
    set_seed(GLOBAL_SEED + worker_id)

def set_rng_state(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(states[3])


def setup_model(cfg):
    net = BDNet(in_channels=config['model']['in_channels'],
                backbone_model=config['model']['backbone_model'], use_edl=cfg.use_edl)
    net = nn.DataParallel(net, device_ids=[0]).cuda()
    model_path = get_path(os.path.join(cfg.checkpoint_path, 'checkpoint-latest.ckpt'))
    net.module.load_state_dict(torch.load(model_path))
    train_path = get_path(os.path.join(cfg.train_state_path, 'checkpoint_latest.ckpt'))
    state_dict = torch.load(train_path)
    set_rng_state(state_dict['state'])
    net.eval()
    return net


def setup_dataloader(cfg):
    train_video_infos = get_video_info(config['dataset']['training']['video_info_path'])
    train_video_annos = get_video_anno(train_video_infos,
                                       config['dataset']['training']['video_anno_path'],
                                       config['dataset']['class_info_path'])
    train_data_dict = load_video_data(train_video_infos,
                                      config['dataset']['training']['video_data_path'])
    train_dataset = THUMOS_Dataset(train_data_dict,
                                   train_video_infos,
                                   train_video_annos)
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                   num_workers=4, worker_init_fn=worker_init_fn,
                                   collate_fn=detection_collate, pin_memory=True, drop_last=True)
    epoch_step_num = len(train_dataset) // cfg.batch_size
    return train_data_loader, epoch_step_num


def compute_iou(pred, target):
    """
    jaccard: A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    """
    pred_left = pred[:, 0]
    pred_right = pred[:, 1]
    target_left = target[:, 0]
    target_right = target[:, 1]

    pred_area = pred_left + pred_right
    target_area = target_left + target_right

    eps = torch.finfo(torch.float32).eps

    inter = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    area_union = target_area + pred_area - inter
    ious = inter / area_union.clamp(min=eps)
    return ious


def get_matched_targets(targets, loc_data, priors, clip_length):
    num_batch = loc_data.size(0)
    num_priors = priors.size(0)
    # match priors and ground truth segments
    loc_t = torch.Tensor(num_batch, num_priors, 2).to(loc_data.device)
    conf_t = torch.LongTensor(num_batch, num_priors).to(loc_data.device)
    prop_loc_t = torch.Tensor(num_batch, num_priors, 2).to(loc_data.device)
    prop_conf_t = torch.LongTensor(num_batch, num_priors).to(loc_data.device)
    for idx in range(num_batch):
        truths = targets[idx][:, :-1]
        labels = targets[idx][:, -1]
        pre_loc = loc_data[idx]
        K = priors.size(0)
        N = truths.size(0)
        center = priors[:, 0].unsqueeze(1).expand(K, N)
        left = (center - truths[:, 0].unsqueeze(0).expand(K, N)) * clip_length
        right = (truths[:, 1].unsqueeze(0).expand(K, N) - center) * clip_length
        area = left + right
        maxn = clip_length * 2
        area[left < 0] = maxn
        area[right < 0] = maxn
        best_truth_area, best_truth_idx = area.min(1)

        loc_t[idx][:, 0] = (priors[:, 0] - truths[best_truth_idx, 0]) * clip_length
        loc_t[idx][:, 1] = (truths[best_truth_idx, 1] - priors[:, 0]) * clip_length
        conf = labels[best_truth_idx]
        conf[best_truth_area >= maxn] = 0
        conf_t[idx] = conf

        iou = compute_iou(pre_loc, loc_t[idx])  # [num_priors]
        prop_conf = conf.clone()
        prop_conf[iou < cfg.overlap_thresh] = 0
        prop_conf_t[idx] = prop_conf
        prop_w = pre_loc[:, 0] + pre_loc[:, 1]
        prop_loc_t[idx][:, 0] = (loc_t[idx][:, 0] - pre_loc[:, 0]) / (0.5 * prop_w)
        prop_loc_t[idx][:, 1] = (loc_t[idx][:, 1] - pre_loc[:, 1]) / (0.5 * prop_w)
    
    return loc_t, conf_t, prop_loc_t, prop_conf_t


def evidence_func(logit, evidence='exp'):
    if evidence == 'relu':
        return F.relu(logit)
    if evidence == 'exp':
        return torch.exp(torch.clamp(logit, -10, 10))
    if evidence == 'softplus':
        return F.softplus(logit)


def grad_edl(logit, target, num_cls=15, evidence='exp'):
    if logit.dim() > 2:
        # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
        logit = logit.view(logit.size(0), logit.size(1), -1)
        logit = logit.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
        logit = logit.view(-1, logit.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
    target = target.view(-1)  # [N,d1,d2,...]->[N*d1*d2*...,]
    # one-hot embedding for the target
    y = torch.eye(num_cls).to(logit.device, non_blocking=True)
    y = y[target]

    pred_alpha = evidence_func(logit, evidence=evidence) + 1  # (alpha = e + 1)
    uncertainty = num_cls / pred_alpha.sum(dim=-1, keepdim=True)  # (N, 1)
    # gradient length
    grad = (1 / pred_alpha - uncertainty) * y  # y_ij * (1/alpha_ij - u_i)
    grad_norm = torch.abs(grad)
    return grad, grad_norm


def get_grad_info(cfg):
    set_seed(cfg.random_seed)
    # Setup model
    net = setup_model(cfg)

    # Setup dataloader
    train_data_loader, epoch_step_num = setup_dataloader(cfg)
    
    # start loop
    all_grad_norms, all_grad_norms_prop = [], []
    all_grads, all_grads_prop = [], []
    with tqdm.tqdm(train_data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (clips, targets, scores, ssl_clips, ssl_targets, flags) in enumerate(pbar):
            clips = clips.cuda()
            targets = [t.cuda() for t in targets]
            with torch.no_grad():
                # run infernece
                output_dict = net(clips, ssl=False)
                # get the matched target
                loc_t, conf_t, prop_loc_t, prop_conf_t = get_matched_targets(targets, output_dict['loc'], output_dict['priors'], cfg.clip_length)
            
            # coarse stage
            conf_p = output_dict['conf'].view(-1, cfg.num_classes)
            targets_conf = conf_t.view(-1, 1)
            if cfg.cls_loss_type == 'focal':
                conf_p = F.softmax(conf_p, dim=1)
            if cfg.os_head:
                inds_keep = targets_conf > 0  # (N,1)
                targets_conf = targets_conf[inds_keep].unsqueeze(-1) - 1  # (M,1), starting from 0
                conf_p = conf_p[inds_keep.squeeze()]  # (M,15)
            if targets_conf.numel() > 0 and cfg.cls_loss_type == 'edl':
                # compute gradient norm (one-hot)
                grads, grad_norms = grad_edl(conf_p, targets_conf, num_cls=cfg.num_classes, evidence=cfg.edl_config['evidence'])

            # refined stage
            prop_conf_p = output_dict['prop_conf'].view(-1, cfg.num_classes)
            prop_conf_t = prop_conf_t.view(-1, 1)
            if cfg.cls_loss_type == 'focal':
                prop_conf_p = F.softmax(prop_conf_p, dim=1)
            if cfg.os_head:
                inds_keep = prop_conf_t > 0  # (N,1)
                prop_conf_t = prop_conf_t[inds_keep].unsqueeze(-1) - 1  # (M,1), starting from 0
                prop_conf_p = prop_conf_p[inds_keep.squeeze()]  # (M,15)
            if prop_conf_t.numel() > 0 and cfg.cls_loss_type == 'edl':
                # compute gradient norm
                grads_prop, grad_norms_prop = grad_edl(prop_conf_p, prop_conf_t, num_cls=cfg.num_classes, evidence=cfg.edl_config['evidence'])

            if grad_norms is not None:
                all_grad_norms.append(grad_norms.cpu().numpy())
                all_grads.append(grads.cpu().numpy())
            if grad_norms_prop is not None:
                all_grad_norms_prop.append(grad_norms_prop.cpu().numpy())
                all_grads_prop.append(grads_prop.cpu().numpy())
    return all_grad_norms, all_grad_norms_prop, all_grads, all_grads_prop

            
def plot_grad_density(save_file, all_grad_norms, num_bins=30, momentum=0.75, fontsize=18):
    
    edges = [float(x) / num_bins for x in range(num_bins+1)]
    edges[-1] += 1e-6
    if momentum > 0:
        acc_sum = [0.0 for _ in range(num_bins)]
    grad_density = np.zeros((num_bins))
    weight_density = np.zeros((num_bins))
    grad_norm = np.concatenate(all_grad_norms, axis=0).sum(axis=-1)  # sum for one-hot grad_norms
    weights = np.zeros_like(grad_norm)
    for i in range(num_bins):
        # compute gradient density (number of gradients in each bin)
        inds = (grad_norm >= edges[i]) & (grad_norm < edges[i+1])
        num_in_bin = inds.sum()
        grad_density[i] = num_in_bin
        # compute the weights
        if num_in_bin > 0:
            if momentum > 0:
                acc_sum[i] = momentum * acc_sum[i] + (1 - momentum) * num_in_bin
                w = 1.0 / acc_sum[i]
            else:
                w = 1.0 / num_in_bin
            weights[inds] = w
            weight_density[i] = w

    # plot
    fig, ax1 = plt.subplots(1,1, figsize=(8,5))
    ax1.plot(edges[:-1], grad_density, 'r-', linewidth=2, label='Grad Density')
    # ax1.set_ylabel('fraction of samples', fontsize=fontsize)
    # ax1.set_yscale('log')
    ax1.legend(fontsize=fontsize, loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(edges[:-1], weight_density, 'b-', linewidth=2, label='weights')
    # ax2.set_ylabel('weights', fontsize=fontsize)
    # ax2.set_yscale('log')
    ax2.legend(fontsize=fontsize, loc='upper right')
    
    plt.xlabel('gradient norm', fontsize=fontsize)
    plt.xlim(0, grad_norm.max() - 1.0/num_bins)
    plt.tight_layout()
    plt.savefig(save_file)
    

def plot_grad_hist(save_file, all_grads, xlim=(-0.1, 0.1), ylim=(0, 100), fontsize=18):
    grads = np.concatenate(all_grads, axis=0).sum(axis=-1)
    fig, ax1 = plt.subplots(1,1, figsize=(8,5))
    plt.hist(grads, 200, density=True, alpha=0.8)
    plt.xlabel("gradient", fontsize=fontsize)
    plt.ylabel("probability density", fontsize=fontsize)
    plt.xlim(xlim) if xlim is not None else [min(grads), max(grads)]
    plt.ylim(ylim) if xlim is not None else None
    plt.tight_layout()
    plt.savefig(save_file)


if __name__ == '__main__':

    cfg = get_basic_config(config)

    output_file = os.path.join(cfg.output_path, 'grad_norms.npz')
    if not os.path.exists(output_file):
        all_grad_norms, all_grad_norms_prop, all_grads, all_grads_prop = get_grad_info(cfg)
        np.savez(output_file[:-4], grad_norms=all_grad_norms, 
                                   grad_norm_prop=all_grad_norms_prop,
                                   grads=all_grads,
                                   grad_prop=all_grads_prop)
    else:
        results = np.load(output_file, allow_pickle=True)
        all_grad_norms, all_grad_norms_prop = results['grad_norms'], results['grad_norm_prop']
        all_grads, all_grads_prop = results['grads'], results['grad_prop']
        print(f'Grad norm result file already exist at {output_file}!')
    
    plot_grad_density(os.path.join(cfg.output_path, 'grad_density.png'), all_grad_norms, num_bins=100, momentum=0.75)
    plot_grad_density(os.path.join(cfg.output_path, 'grad_prop_density.png'), all_grad_norms_prop, num_bins=100, momentum=0.75)

    plot_grad_hist(os.path.join(cfg.output_path, 'grad_hist.png'), all_grads, xlim=(-0.1,0.1), ylim=(0,100))
    plot_grad_hist(os.path.join(cfg.output_path, 'grad_prop_hist.png'), all_grads_prop, xlim=None, ylim=None)
    