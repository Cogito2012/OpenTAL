# This code is originally from the official ActivityNet repo
# https://github.com/activitynet/ActivityNet

import json
import urllib.request
import os, pickle
import matplotlib.pyplot as plt
import numpy as np

API = 'http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/challenge17/api.py'


def get_blocked_videos(api=API):
    api_url = '{}?action=get_blocked'.format(api)
    req = urllib.request.Request(api_url)
    response = urllib.request.urlopen(req)
    return json.loads(response.read().decode('utf-8'))


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def wrapper_segment_iou(target_segments, candidate_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    candidate_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [n x m] with IOU ratio.
    Note: It assumes that candidate-segments are more scarce that target-segments
    """
    if candidate_segments.ndim != 2 or target_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    n, m = candidate_segments.shape[0], target_segments.shape[0]
    tiou = np.empty((n, m))
    for i in xrange(m):
        tiou[:, i] = segment_iou(target_segments[i, :], candidate_segments)

    return tiou


def open_set_detection_rate(preds, pred_cls, gt_cls):
    """ preds: ndarray, (N,), [0, 1]
        pred_cls: ndarray, (N,), >0: known
        gt_cls: ndarray, (N,), 0: unknown, >0: known
    """
    
    x1, x2 = preds[gt_cls > 0], preds[gt_cls == 0]  # known preds & unknown preds
    m_x1 = np.zeros(len(x1))
    m_x1[pred_cls[gt_cls > 0] == gt_cls[gt_cls > 0]] = 1  # for known preds, fraction of correct cls
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)  # target of correct known (in all preds)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)  # target of all unknown (in all preds)
    predict = np.concatenate((x1, x2), axis=0)  # re-organize pred score
    n = len(preds)

    # Cutoffs are of prediction values
    CCR = [0 for x in range(n+2)]
    FPR = [0 for x in range(n+2)]

    # sort the targets by ascending order of predictions
    idx = predict.argsort()
    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    # for each cut-off score
    for k in range(n-1):
        # accumulated sum
        CC = s_k_target[k+1:].sum()
        FP = s_u_target[k:].sum()
        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1)) if len(x1) > 0 else 1.0  # fraction of correct classification in known preds
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2)) if len(x2) > 0 else 0.0 # fraction of unknown preds that are classified as any known
    
    # extreme cases
    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n+1] = 1.0
    FPR[n+1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)  # descending order by FPR
    OSCR = 0
    # Compute AUROC Using Trapezoidal Rule
    for j in range(n+1):
        w =   ROC[j][0] - ROC[j+1][0]  # delta_FPR
        h =  (ROC[j][1] + ROC[j+1][1]) / 2.0  # mean_CCR
        OSCR = OSCR + h*w  # area under the curve
    return OSCR, FPR, CCR


def save_curve_data(roc_data, pr_data, save_path, vis=False, fontsize=18):
    os.makedirs(save_path, exist_ok=True)
    # save roc data
    with open(os.path.join(save_path, 'roc_data.pkl'), 'wb') as f:
        pickle.dump(roc_data, f, pickle.HIGHEST_PROTOCOL)
    # save pr data
    with open(os.path.join(save_path, 'pr_data.pkl'), 'wb') as f:
        pickle.dump(pr_data, f, pickle.HIGHEST_PROTOCOL)
    # draw curves
    if vis:
        line_styles = ['r-', 'c-', 'g-', 'b-', 'k']
        # plot roc curve
        plt.figure(figsize=(8, 5))
        for tidx, (fpr, tpr, auc, tiou) in enumerate(zip(roc_data['fpr'], roc_data['tpr'], roc_data['auc'], roc_data['tiou'])):
            plt.plot(fpr, tpr, line_styles[tidx], label=f'tIoU={tiou}, auc={auc*100:.2f}%')
        plt.xlabel('False Positive Rate', fontsize=fontsize)
        plt.ylabel('True Positive Rate', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'AUC_ROC.png'))
        plt.close()
        # plot pr curve
        plt.figure(figsize=(8, 5))
        for tidx, (precision, recall, auc, tiou) in enumerate(zip(pr_data['precision'], pr_data['recall'], pr_data['auc'], pr_data['tiou'])):
            plt.plot(recall, precision, line_styles[tidx], label=f'tIoU={tiou}, auc={auc*100:.2f}%')
        plt.xlabel('Recall', fontsize=fontsize)
        plt.ylabel('Precision', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'AUC_PR.png'))
        plt.close()


def save_curve_osdr_data(osdr_data, save_path, vis=False, fontsize=18):
    os.makedirs(save_path, exist_ok=True)
    # save roc data
    with open(os.path.join(save_path, 'osdr_data.pkl'), 'wb') as f:
        pickle.dump(osdr_data, f, pickle.HIGHEST_PROTOCOL)
    # draw curves
    if vis:
        line_styles = ['r-', 'c-', 'g-', 'b-', 'k']
        # plot roc curve
        plt.figure(figsize=(8, 5))
        for tidx, (fpr, cdr, osdr, tiou) in enumerate(zip(osdr_data['fpr'], osdr_data['cdr'], osdr_data['osdr'], osdr_data['tiou'])):
            plt.plot(fpr[:-2], cdr[:-2], line_styles[tidx], label=f'tIoU={tiou}, auc={osdr*100:.2f}%')
        plt.xlabel('False Positive Rate', fontsize=fontsize)
        plt.ylabel('Correct Detection Rate', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'AUC_OSDR.png'))
        plt.close()