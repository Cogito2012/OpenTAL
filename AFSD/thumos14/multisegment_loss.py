import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from AFSD.common.config import config
from .cls_loss import FocalLoss_Ori, EvidenceLoss


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def iou_loss(pred, target, weight=None, loss_type='giou', reduction='none'):
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

    if loss_type == 'linear_iou':
        loss = 1.0 - ious
    elif loss_type == 'giou':
        ac_uion = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1.0 - gious
    else:
        loss = ious

    if weight is not None:
        loss = loss * weight.view(loss.size())
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        loss = loss.mean()
    return loss


def calc_ioa(pred, target):
    pred_left = pred[:, 0]
    pred_right = pred[:, 1]
    target_left = target[:, 0]
    target_right = target[:, 1]

    pred_area = pred_left + pred_right
    eps = torch.finfo(torch.float32).eps

    inter = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    ioa = inter / pred_area.clamp(min=eps)
    return ioa


class MultiSegmentLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, negpos_ratio, use_gpu=True, 
                 cls_loss_type='focal', edl_config=None):
        super(MultiSegmentLoss, self).__init__()
        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.negpos_ratio = negpos_ratio
        self.use_gpu = use_gpu
        self.cls_loss_type = cls_loss_type
        if self.cls_loss_type == 'focal':
            self.cls_loss = FocalLoss_Ori(num_classes, balance_index=0, size_average=False, alpha=0.25)
        elif self.cls_loss_type == 'edl':
            self.cls_loss = EvidenceLoss(num_classes, edl_config)
        self.center_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets, pre_locs=None):
        """
        :param predictions: a tuple containing loc, conf and priors
        :param targets: ground truth segments and labels
        :return: loc loss and conf loss
        """
        loc_data, conf_data, \
        prop_loc_data, prop_conf_data, center_data, priors = predictions
        num_batch = loc_data.size(0)
        num_priors = priors.size(0)
        num_classes = self.num_classes
        clip_length = config['dataset']['training']['clip_length']
        # match priors and ground truth segments
        loc_t = torch.Tensor(num_batch, num_priors, 2).to(loc_data.device)
        conf_t = torch.LongTensor(num_batch, num_priors).to(loc_data.device)
        prop_loc_t = torch.Tensor(num_batch, num_priors, 2).to(loc_data.device)
        prop_conf_t = torch.LongTensor(num_batch, num_priors).to(loc_data.device)

        with torch.no_grad():
            for idx in range(num_batch):
                truths = targets[idx][:, :-1]
                labels = targets[idx][:, -1]
                pre_loc = loc_data[idx]
                """
                match gt
                """
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

                iou = iou_loss(pre_loc, loc_t[idx], loss_type='calc iou')  # [num_priors]
                prop_conf = conf.clone()
                prop_conf[iou < self.overlap_thresh] = 0
                prop_conf_t[idx] = prop_conf
                prop_w = pre_loc[:, 0] + pre_loc[:, 1]
                prop_loc_t[idx][:, 0] = (loc_t[idx][:, 0] - pre_loc[:, 0]) / (0.5 * prop_w)
                prop_loc_t[idx][:, 1] = (loc_t[idx][:, 1] - pre_loc[:, 1]) / (0.5 * prop_w)

        pos = conf_t > 0  # [num_batch, num_priors]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # [num_batch, num_priors, 2]
        gt_loc_t = loc_t.clone()
        loc_p = loc_data[pos_idx].view(-1, 2)
        loc_target = loc_t[pos_idx].view(-1, 2)
        if loc_p.numel() > 0:
            loss_l = iou_loss(loc_p, loc_target, loss_type='giou', reduction='sum')

        else:
            loss_l = loc_p.sum()

        prop_pos = prop_conf_t > 0
        prop_pos_idx = prop_pos.unsqueeze(-1).expand_as(prop_loc_data)  # [num_batch, num_priors, 2]
        prop_loc_p = prop_loc_data[prop_pos_idx].view(-1, 2)
        prop_loc_t = prop_loc_t[prop_pos_idx].view(-1, 2)

        if prop_loc_p.numel() > 0:
            loss_prop_l = F.l1_loss(prop_loc_p, prop_loc_t, reduction='sum')
        else:
            loss_prop_l = prop_loc_p.sum()

        prop_pre_loc = loc_data[pos_idx].view(-1, 2)
        cur_loc_t = gt_loc_t[pos_idx].view(-1, 2)
        prop_loc_p = prop_loc_data[pos_idx].view(-1, 2)
        center_p = center_data[pos.unsqueeze(pos.dim())].view(-1)
        if prop_pre_loc.numel() > 0:
            prop_pre_w = (prop_pre_loc[:, 0] + prop_pre_loc[:, 1]).unsqueeze(-1)
            cur_loc_p = 0.5 * prop_pre_w * prop_loc_p + prop_pre_loc
            ious = iou_loss(cur_loc_p, cur_loc_t, loss_type='calc iou').clamp_(min=0)
            loss_ct = F.binary_cross_entropy_with_logits(
                center_p,
                ious,
                reduction='sum'
            )
        else:
            loss_ct = prop_pre_loc.sum()

        # softmax focal loss
        conf_p = conf_data.view(-1, num_classes)
        targets_conf = conf_t.view(-1, 1)
        if self.cls_loss_type == 'focal':
            conf_p = F.softmax(conf_p, dim=1)
        loss_c = self.cls_loss(conf_p, targets_conf)

        prop_conf_p = prop_conf_data.view(-1, num_classes)
        if self.cls_loss_type == 'focal':
            prop_conf_p = F.softmax(prop_conf_p, dim=1)
        loss_prop_c = self.cls_loss(prop_conf_p, prop_conf_t)

        N = max(pos.sum(), 1)
        PN = max(prop_pos.sum(), 1)
        loss_l /= N
        loss_c /= N
        loss_prop_l /= PN
        loss_prop_c /= PN
        loss_ct /= N
        # print(N, num_neg.sum())
        return loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct
