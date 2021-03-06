import torch
import torch.nn as nn
import torch.nn.functional as F
from AFSD.common.config import config
from .cls_loss import FocalLoss_Ori, EvidenceLoss, ActionnessLoss


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


bounds = [[0, 30], [15, 60], [30, 120], [60, 240], [96, 768], [256, 768]]
prior_lb = None
prior_rb = None


def gen_bounds(priors):
    global prior_lb, prior_rb
    K = priors.size(0)
    prior_lb = priors[:, 1].clone()
    prior_rb = priors[:, 1].clone()
    for i in range(K):
        prior_lb[i] = bounds[int(prior_lb[i])][0]
        prior_rb[i] = bounds[int(prior_rb[i])][1]
    prior_lb = prior_lb.unsqueeze(1)
    prior_rb = prior_rb.unsqueeze(1)


class MultiSegmentLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, negpos_ratio, use_gpu=True,
                 cls_loss_type='focal', edl_config=None, os_head=False, size_average=False):
        super(MultiSegmentLoss, self).__init__()
        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.negpos_ratio = negpos_ratio
        self.use_gpu = use_gpu
        self.cls_loss_type = cls_loss_type
        if self.cls_loss_type == 'focal':
            self.cls_loss = FocalLoss_Ori(num_classes, balance_index=0, size_average=size_average, alpha=0.25)
        elif self.cls_loss_type == 'edl':
            self.cls_loss = EvidenceLoss(num_classes, edl_config, size_average=size_average)
        self.iou_aware = True if self.cls_loss_type == 'edl' and self.cls_loss.iou_aware else False
        self.os_head = os_head
        if self.os_head:
            self.act_loss = ActionnessLoss(size_average=size_average, weight=0.1)
        self.size_average = size_average
        self.center_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets, pre_locs=None):
        """
        :param predictions: a tuple containing loc, conf and priors
        :param targets: ground truth segments and labels
        :return: loc loss and conf loss
        """
        loc_data, conf_data, \
        prop_loc_data, prop_conf_data, center_data, priors, act_data, prop_act_data = predictions
        # priors = priors[0]
        num_batch = loc_data.size(0)
        num_priors = priors.size(0)
        num_classes = self.num_classes
        clip_length = config['dataset']['training']['clip_length']

        loss_l_list = []
        loss_c_list = []
        loss_ct_list = []
        loss_prop_l_list = []
        loss_prop_c_list = []
        loss_act_list = []
        loss_prop_act_list = []

        iou_pred = torch.Tensor(num_priors, num_batch).to(loc_data.device) if self.iou_aware else None
        for idx in range(num_batch):
            loc_t = torch.Tensor(num_priors, 2).to(loc_data.device)
            conf_t = torch.LongTensor(num_priors).to(loc_data.device)
            prop_loc_t = torch.Tensor(num_priors, 2).to(loc_data.device)
            prop_conf_t = torch.LongTensor(num_priors).to(loc_data.device)

            loc_p = loc_data[idx]
            logit_p = conf_data[idx]
            prop_loc_p = prop_loc_data[idx]
            prop_logit_p = prop_conf_data[idx]
            center_p = center_data[idx]
            if self.os_head:
                act_p = act_data[idx]
                prop_act_p = prop_act_data[idx]

            with torch.no_grad():
                # match priors and ground truth segments
                truths = targets[idx][:, :-1]
                labels = targets[idx][:, -1]
                """
                match gt
                """
                K = priors.size(0)
                N = truths.size(0)
                center = priors[:, 0].unsqueeze(1).expand(K, N)
                left = (center - truths[:, 0].unsqueeze(0).expand(K, N)) * clip_length
                right = (truths[:, 1].unsqueeze(0).expand(K, N) - center) * clip_length
                max_dis = torch.max(left, right)
                if prior_lb is None or prior_rb is None:
                    gen_bounds(priors)
                l_bound = prior_lb.expand(K, N)
                r_bound = prior_rb.expand(K, N)
                area = left + right
                maxn = clip_length * 2
                area[left < 0] = maxn
                area[right < 0] = maxn
                area[max_dis <= l_bound] = maxn
                area[max_dis > r_bound] = maxn
                best_truth_area, best_truth_idx = area.min(1)

                loc_t[:, 0] = (priors[:, 0] - truths[best_truth_idx, 0]) * clip_length
                loc_t[:, 1] = (truths[best_truth_idx, 1] - priors[:, 0]) * clip_length
                conf = labels[best_truth_idx]
                conf[best_truth_area >= maxn] = 0
                conf_t[:] = conf

                iou = iou_loss(loc_p, loc_t, loss_type='calc iou')  # [num_priors]
                if self.iou_aware:
                    iou_pred[:, idx] = iou
                if (conf > 0).sum() > 0:
                    max_iou, max_iou_idx = iou[conf > 0].max(0)
                else:
                    max_iou = 2.0
                # print(max_iou)
                prop_conf = conf.clone()
                prop_conf[iou < min(self.overlap_thresh, max_iou)] = 0
                prop_conf_t[:] = prop_conf
                prop_w = loc_p[:, 0] + loc_p[:, 1]
                prop_loc_t[:, 0] = (loc_t[:, 0] - loc_p[:, 0]) / (0.5 * prop_w)
                prop_loc_t[:, 1] = (loc_t[:, 1] - loc_p[:, 1]) / (0.5 * prop_w)

            pos = conf_t > 0  # [num_priors]
            pos_idx = pos.unsqueeze(-1).expand_as(loc_p)  # [num_priors, 2]
            gt_loc_t = loc_t.clone()
            loc_p = loc_p[pos_idx].view(-1, 2)
            loc_target = loc_t[pos_idx].view(-1, 2)
            if loc_p.numel() > 0:
                loss_l = iou_loss(loc_p, loc_target, loss_type='giou', reduction='sum')
            else:
                loss_l = loc_p.sum()

            prop_pos = prop_conf_t > 0
            prop_pos_idx = prop_pos.unsqueeze(-1).expand_as(prop_loc_p)  # [num_priors, 2]
            target_prop_loc_p = prop_loc_p[prop_pos_idx].view(-1, 2)
            prop_loc_t = prop_loc_t[prop_pos_idx].view(-1, 2)

            if prop_loc_p.numel() > 0:
                loss_prop_l = F.smooth_l1_loss(target_prop_loc_p, prop_loc_t, reduction='sum')
            else:
                loss_prop_l = target_prop_loc_p.sum()

            prop_pre_loc = loc_p
            cur_loc_t = gt_loc_t[pos_idx].view(-1, 2)
            prop_loc_p = prop_loc_p[pos_idx].view(-1, 2)
            center_p = center_p[pos.unsqueeze(-1)].view(-1)
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

            # classification loss in the coarse stage
            conf_p = logit_p.view(-1, num_classes)
            targets_conf = conf_t.view(-1, 1)
            if self.cls_loss_type == 'focal':
                conf_p = F.softmax(conf_p, dim=1)
            if self.os_head:
                inds_keep = targets_conf > 0  # (N,1)
                targets_conf = targets_conf[inds_keep].unsqueeze(-1) - 1  # (M,1), starting from 0
                conf_p = conf_p[inds_keep.squeeze()]  # (M,15)
            if targets_conf.numel() > 0:
                loss_c = self.cls_loss(conf_p, targets_conf)
            else:  # empty, do not need to compute loss
                loss_c = torch.tensor(0.0).to(conf_p.device)

            if self.os_head:
                act_scores = act_p.view(-1, 1)  # [N, 1]
                act_targets = inds_keep.to(torch.float32)  # [N, 1]
                loss_act, AN = self.act_loss(act_scores, act_targets)

            # classification loss in the refined stage
            prop_conf_p = prop_logit_p.view(-1, num_classes)
            prop_conf_t = prop_conf_t.view(-1, 1)
            if self.cls_loss_type == 'focal':
                prop_conf_p = F.softmax(prop_conf_p, dim=1)
            if self.os_head:
                inds_keep = prop_conf_t > 0  # (N,1)
                prop_conf_t = prop_conf_t[inds_keep].unsqueeze(-1) - 1  # (M,1), starting from 0
                prop_conf_p = prop_conf_p[inds_keep.squeeze()]  # (M,15)
            if prop_conf_t.numel() > 0:
                loss_prop_c = self.cls_loss(prop_conf_p, prop_conf_t)
            else:  # empty, do not need to compute loss
                loss_prop_c = torch.tensor(0.0).to(prop_conf_p.device)

            if self.iou_aware:
                logit_pred = prop_conf_data[idx].view(-1, num_classes)
                loss_iouc = self.cls_loss.iou_calib(logit_pred, iou_pred[:, idx].view(-1), mean=True)

            if self.os_head:
                prop_act_scores = prop_act_p.view(-1, 1)  # [N, 1]
                prop_act_targets = inds_keep.to(torch.float32)  # [N, 1]
                loss_prop_act, PAN = self.act_loss(prop_act_scores, prop_act_targets)

            N = max(pos.sum(), 1)
            PN = max(prop_pos.sum(), 1)
            loss_l /= N if not self.size_average else loss_l
            loss_c /= N if not self.size_average else loss_c
            loss_prop_l /= PN if not self.size_average else loss_prop_l
            loss_prop_c /= PN if not self.size_average else loss_prop_c
            if self.iou_aware:
                loss_prop_c += loss_iouc
            loss_ct /= N if not self.size_average else loss_ct
            if self.os_head:
                loss_act = loss_act / AN if not self.size_average else loss_act
                loss_prop_act = loss_prop_act / PAN if not self.size_average else loss_prop_act
                loss_act_list.append(loss_act)
                loss_prop_act_list.append(loss_prop_act)

            loss_l_list.append(loss_l)
            loss_c_list.append(loss_c)
            loss_prop_l_list.append(loss_prop_l)
            loss_prop_c_list.append(loss_prop_c)
            loss_ct_list.append(loss_ct)

        # print(N, num_neg.sum())
        loss_l = sum(loss_l_list) / num_batch
        loss_c = sum(loss_c_list) / num_batch
        loss_ct = sum(loss_ct_list) / num_batch
        loss_prop_l = sum(loss_prop_l_list) / num_batch
        loss_prop_c = sum(loss_prop_c_list) / num_batch
        if self.os_head:
            loss_act = sum(loss_act_list) / num_batch
            loss_prop_act = sum(loss_prop_act_list) / num_batch
        else:
            loss_act, loss_prop_act = None, None
        
        return loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct, loss_act, loss_prop_act
