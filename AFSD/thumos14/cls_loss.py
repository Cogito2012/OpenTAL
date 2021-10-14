import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, size_average=True):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float, int)):
            assert 0 < self.alpha < 1.0, 'alpha should be in `(0,1)`)'
            assert balance_index > -1
            alpha = torch.ones((self.num_class))
            alpha *= 1 - self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target):
        """ logit: softmax scores (N, K+1)
            target: integeral labels (N, 1) \in [0,...,K]
        """
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]

        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        pt = logit.gather(1, target).view(-1) + self.eps  # avoid apply, (N,)
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            self.alpha = self.alpha.to(logpt.device)  # (K+1,)

        alpha_class = self.alpha.gather(0, target.view(-1))  # (N,)
        logpt = alpha_class * logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class EvidenceLoss(nn.Module):
    def __init__(self, num_cls, cfg, size_average=False):
        super(EvidenceLoss, self).__init__()
        self.num_cls = num_cls
        self.loss_type = cfg['loss_type']
        self.evidence = cfg['evidence']
        self.with_focal = cfg['with_focal']
        self.with_ghm = cfg['with_ghm'] if 'with_ghm' in cfg else False
        self.eps = 1e-10
        if self.with_focal:
            alpha = torch.ones((self.num_cls)) * (1 - cfg['alpha'])  # foreground class
            alpha[0] = cfg['alpha']  # background class
            self.alpha = alpha
            self.gamma = cfg['gamma']
        if self.with_ghm:
            self.num_bins = cfg['num_bins']
            self.momentum = cfg['momentum']
            self.edges = [float(x) / self.num_bins for x in range(self.num_bins+1)]
            self.edges[-1] += 1e-6
            if self.momentum > 0:
                self.acc_sum = [0.0 for _ in range(self.num_bins)]
        self.size_average = size_average
    

    def forward(self, logit, target):
        """ logit, shape=(N, K+1)
            target, shape=(N, 1)
        """
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1)  # [N,d1,d2,...]->[N*d1*d2*...,]

        out_dict = dict()

        # one-hot embedding for the target
        y = torch.eye(self.num_cls).to(logit.device, non_blocking=True)
        y = y[target]  # (N, K+1)
        
        # get loss func
        loss, func = self.get_loss_func()

        # compute losses
        pred_alpha = self.evidence_func(logit) + 1  # (alpha = e + 1)
        loss_out = loss(y, pred_alpha, func=func, target=target)
        out_dict.update(loss_out)

        # accumulate total loss
        total_loss = 0
        for k, v in loss_out.items():
            if 'loss' in k:
                total_loss += v
        out_dict.update({'total_loss': total_loss})
        return total_loss


    def get_loss_func(self):
        if self.loss_type == 'mse':
            return self.mse_loss, None
        elif self.loss_type == 'log':
            return self.edl_loss, torch.log
        elif self.loss_type == 'digamma':
            return self.edl_loss, torch.digamma
        else:
            raise NotImplementedError


    def evidence_func(self, logit):
        if self.evidence == 'relu':
            return F.relu(logit)

        if self.evidence == 'exp':
            return torch.exp(torch.clamp(logit, -10, 10))

        if self.evidence == 'softplus':
            return F.softplus(logit)
        

    def mse_loss(self, y, alpha, func=None, target=None):
        """Used only for loss_type == 'mse'
        y: the one-hot labels (batchsize, num_classes)
        alpha: the predictions (batchsize, num_classes)
        annealing_coef: dependent on training epoch
        """
        losses = {}
        # compute loss by considering the temporal penalty
        loglikelihood_err, loglikelihood_var = self.loglikelihood_loss(y, alpha)
        if self.size_average:
            loglikelihood_err = torch.mean(loglikelihood_err)
            loglikelihood_var = torch.mean(loglikelihood_var)
        else:
            loglikelihood_err = torch.sum(loglikelihood_err)
            loglikelihood_var = torch.sum(loglikelihood_var)
        losses.update({'cls_loss': loglikelihood_err, 'var_loss': loglikelihood_var})
        return losses


    def edl_loss(self, y, alpha, func=torch.log, target=None):
        """Used for both loss_type == 'log' and loss_type == 'digamma'
        y: the one-hot labels (batchsize, num_classes)
        alpha: the predictions (batchsize, num_classes)
        annealing_coef: dependent on training epoch
        func: function handler (torch.log, or torch.digamma)
        """
        losses = {}
        S = torch.sum(alpha, dim=1, keepdim=True)  # (B, 1)
        if self.with_focal:
            if self.alpha.device != alpha.device:
                self.alpha = self.alpha.to(alpha.device)
            pred_scores, pred_cls = torch.max(alpha / S, 1)
            alpha_class = self.alpha.gather(0, target.view(-1))  # (N,)
            weight = alpha_class * torch.pow(torch.sub(1.0, pred_scores), self.gamma)  # (N,)
            cls_loss = torch.sum(y * weight.unsqueeze(-1) * (func(S) - func(alpha)), dim=1)
        elif self.with_ghm:
            alpha_pred = alpha.detach().clone()  # (N, K)
            uncertainty = self.num_cls / alpha_pred.sum(dim=-1, keepdim=True)  # (N, 1)
            weights = torch.pow(1 / alpha_pred - uncertainty, 2)
            # gradient length
            # grad_norm = torch.abs(1 / alpha_pred - uncertainty) * y  # y_ij * (1/alpha_ij - u_i)
            # n = 0  # n valid bins
            # weights = torch.zeros_like(alpha)
            # for i in range(self.num_bins):
            #     inds = (grad_norm >= self.edges[i]) & (grad_norm < self.edges[i+1])
            #     num_in_bin = inds.sum().item()
            #     if num_in_bin > 0:
            #         if self.momentum > 0:
            #             self.acc_sum[i] = self.momentum * self.acc_sum[i] \
            #                 + (1 - self.momentum) * num_in_bin
            #             weights[inds] = 1.0 / self.acc_sum[i]
            #         else:
            #             weights[inds] = 1.0 / num_in_bin
            #         n += 1
            # if n > 0:
            #     weights = weights / n
            # compute the weighted EDL loss
            cls_loss = torch.sum(y * weights * (func(S) - func(alpha)), dim=1)
        else:
            cls_loss = torch.sum(y * (func(S) - func(alpha)), dim=1)
        if self.size_average:
            cls_loss = torch.mean(cls_loss)
        else:
            cls_loss = torch.sum(cls_loss)
        losses.update({'cls_loss': cls_loss})
        return losses


    def loglikelihood_loss(self, y, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        return loglikelihood_err, loglikelihood_var


class ActionnessLoss(nn.Module):
    def __init__(self, size_average=False, weight=0.1, margin=1.0):
        super(ActionnessLoss, self).__init__()
        self.size_average = size_average
        self.weight = weight
        self.margin = margin

    def forward(self, logit, target):
        """ logit, shape=(N, 1), unbounded logits
            target, shape=(N, 1) bianry values
        """
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        label = target.view(-1)  # [N,d1,d2,...]->[N*d1*d2*...,]
        pred = logit.view(-1) if logit.size(-1) == 1 else logit

        # split the predictions into positive and negative setss
        pos_pred, pos_label = pred[label > 0], label[label > 0]
        neg_pred, neg_label = pred[label == 0], label[label == 0]

        num_pos = pos_pred.numel()
        num_neg = neg_pred.numel()
        topM = min(num_pos, num_neg) - 1  # reserve one for rank loss
        if topM > 0: # both pos and neg sets have at least 2 samples
            _, inds = neg_pred.sort()  # by default, it is ascending sort
            # select the top-M negatives
            neg_clean_pred = neg_pred[inds[:topM]]
            neg_clean_label = neg_label[inds[:topM]]
            pred = torch.cat((pos_pred, neg_clean_pred), dim=0)
            label = torch.cat((pos_label, neg_clean_label), dim=0)
            num_neg = topM
        
        # compute BCE loss
        reduction = 'mean' if self.size_average else 'sum'
        loss_bce = F.binary_cross_entropy_with_logits(pred, label, reduction=reduction)

        # compute rank loss
        loss_rank = 0
        if topM > 0:
            neg_noisy_pred, _ = torch.max(neg_pred, dim=0)
            pos_clean_pred, _ = torch.max(pos_pred, dim=0)
            loss_rank = torch.maximum(torch.tensor(0.0).to(pred.device), self.margin - neg_noisy_pred + pos_clean_pred.detach())

        loss_total = loss_bce + self.weight * loss_rank
        return loss_total, num_pos + num_neg