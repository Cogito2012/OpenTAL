import torch
import torch.nn as nn
import numpy as np
import scipy.spatial.distance as spd


def compute_eucos_dist(mav, feature):
    dist = spd.euclidean(mav, feature) / 200 + spd.cosine(mav, feature)
    return dist


class OpenMax(nn.Module):
    def __init__(self, weibull_model, rank=1):
        super(OpenMax, self).__init__()
        self.weibull_model = weibull_model
        self.class_names = list(weibull_model.keys())
        self.num_cls = len(self.class_names)
        self.rank = rank


    def compute_openmax_prob(self, openmax_score, openmax_score_u):
        prob_scores, prob_unknowns = [], []
        channel_scores, channel_unknowns = [], []
        for gt_cls in range(self.num_cls):
            channel_scores += [np.exp(openmax_score[gt_cls])]
        
        total_denominator = np.sum(np.exp(openmax_score)) + np.exp(np.sum(openmax_score_u))
        prob_scores += [channel_scores/total_denominator]
        prob_unknowns += [np.exp(np.sum(openmax_score_u))/total_denominator]
            
        prob_scores = np.array(prob_scores)
        prob_unknowns = np.array(prob_unknowns)

        scores = np.mean(prob_scores, axis=0)
        unknowns = np.mean(prob_unknowns, axis=0)
        modified_scores =  [unknowns] + scores.tolist()  # the first one is unknown
        assert len(modified_scores) == self.num_cls + 1
        modified_scores = np.array(modified_scores)
        return modified_scores


    def openmax_recalibrate(self, logit, feat):
        """ logit: (K,)
            feat: (D,)
        """
        # get the ranked alpha
        alpharank = min(self.num_cls, self.rank)
        ranked_list = logit.argsort().ravel()[::-1]
        alpha_weights = [((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]
        ranked_alpha = np.zeros((self.num_cls,))
        for i in range(len(alpha_weights)):
            ranked_alpha[ranked_list[i]] = alpha_weights[i]

        openmax_channel = []
        openmax_unknown = []
        for cls_gt in range(self.num_cls):
            # get distance between current channel and mean vector
            cls_name = self.class_names[cls_gt]
            mav_train = self.weibull_model[cls_name]['mean_vec']
            category_weibull = self.weibull_model[cls_name]['model']

            channel_distance = compute_eucos_dist(mav_train, feat)
            # obtain w_score for the distance and compute probability of the distance
            wscore = category_weibull[0].w_score(channel_distance)
            modified_score = logit[cls_gt] * ( 1 - wscore*ranked_alpha[cls_gt] )
            openmax_channel += [modified_score]
            openmax_unknown += [logit[cls_gt] - modified_score]

        openmax_score = np.array(openmax_channel)
        openmax_score_u = np.array(openmax_unknown)
        # Pass the recalibrated scores into openmax
        openmax_prob = self.compute_openmax_prob(openmax_score, openmax_score_u)
        return openmax_prob
        

    def forward(self, logits_in, feature_in):
        """ logits: (N, K)
            feature: (N, D)
        """
        logits = logits_in.cpu().numpy()
        feature = feature_in.cpu().numpy()
        openmax_probs = np.zeros((logits.shape[0], self.num_cls + 1))
        for i, (logit, feat) in enumerate(zip(logits, feature)):
            openmax_probs[i] = self.openmax_recalibrate(logit, feat)
        openmax_probs = torch.from_numpy(openmax_probs).to(logits_in.device)
        return openmax_probs
