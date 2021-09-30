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
        pt = logit.gather(1, target).view(-1) + self.eps  # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            self.alpha = self.alpha.to(logpt.device)

        alpha_class = self.alpha.gather(0, target.view(-1))
        logpt = alpha_class * logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class EvidenceLoss(nn.Module):
    def __init__(self, num_cls, cfg):
        super(EvidenceLoss, self).__init__()
        self.num_cls = num_cls
        self.with_kldiv = cfg['with_kldiv']
        self.annealing_method = cfg['annealing']
        self.annealing_start = cfg['anneal_start']
        self.annealing_step = cfg['anneal_step']
        self.loss_type = cfg['loss_type']
        self.evidence = cfg['evidence']
        self.eps = 1e-10
        self.train_status = {'epoch': 0, 'total_epoch': 0}

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
        annealing_coef = 0.0
        if self.with_kldiv:
            # compute annealing coefficient 
            annealing_coef = self.compute_annealing_coef(self.train_status['epoch'], self.train_status['total_epoch'])
            out_dict.update({'annealing_coef': annealing_coef})

        # one-hot embedding for the target
        y = torch.eye(self.num_cls).to(logit.device, non_blocking=True)
        y = y[target]  # (N, K+1)
        
        # get loss func
        loss, func = self.get_loss_func()

        # compute losses
        pred_alpha = self.evidence_func(logit) + 1  # (alpha = e + 1)
        loss_out = loss(y, pred_alpha, annealing_coef=annealing_coef, func=func, target=target)
        out_dict.update(loss_out)

        # accumulate total loss
        total_loss = 0
        for k, v in loss_out.items():
            if 'loss' in k:
                total_loss += v
        out_dict.update({'total_loss': total_loss})
        return total_loss



    def compute_annealing_coef(self, epoch_num, total_epoch):
        # annealing coefficient
        if self.annealing_method == 'step':
            annealing_coef = torch.min(torch.tensor(
                1.0, dtype=torch.float32), torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32))
        elif self.annealing_method == 'exp':
            annealing_start = torch.tensor(self.annealing_start, dtype=torch.float32)
            annealing_coef = annealing_start * torch.exp(-torch.log(annealing_start) / total_epoch * epoch_num)
        else:
            raise NotImplementedError
        return annealing_coef


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
        

    def mse_loss(self, y, alpha, annealing_coef=0, func=None, target=None):
        """Used only for loss_type == 'mse'
        y: the one-hot labels (batchsize, num_classes)
        alpha: the predictions (batchsize, num_classes)
        annealing_coef: dependent on training epoch
        """
        losses = {}
        # compute loss by considering the temporal penalty
        loglikelihood_err, loglikelihood_var = self.loglikelihood_loss(y, alpha)
        losses.update({'cls_loss': loglikelihood_err, 'var_loss': loglikelihood_var})

        if self.with_kldiv:
            kl_alpha = (alpha - 1) * (1 - y) + 1
            kl_div = torch.mean(annealing_coef * self.kl_divergence(kl_alpha))
            losses.update({'kl_loss': kl_div})
        return losses


    def edl_loss(self, y, alpha, annealing_coef=0, func=torch.log, target=None):
        """Used for both loss_type == 'log' and loss_type == 'digamma'
        y: the one-hot labels (batchsize, num_classes)
        alpha: the predictions (batchsize, num_classes)
        annealing_coef: dependent on training epoch
        func: function handler (torch.log, or torch.digamma)
        """
        losses = {}
        S = torch.sum(alpha, dim=1, keepdim=True)  # (B, 1)
        A = torch.mean(torch.sum(y * (func(S) - func(alpha)), dim=1))
        losses.update({'cls_loss': A})

        if self.with_kldiv:
            kl_alpha = (alpha - 1) * (1 - y) + 1
            kl_div = torch.mean(annealing_coef * self.kl_divergence(kl_alpha))
            losses.update({'kl_loss': kl_div})
        return losses


    def loglikelihood_loss(self, y, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.mean(torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True))
        loglikelihood_var = torch.mean(torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True))
        return loglikelihood_err, loglikelihood_var


    def kl_divergence(self, alpha):
        beta = torch.ones([1, self.num_cls], dtype=torch.float32).to(alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - \
            torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                            keepdim=True) - torch.lgamma(S_beta)

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                    keepdim=True) + lnB + lnB_uni
        return kl
