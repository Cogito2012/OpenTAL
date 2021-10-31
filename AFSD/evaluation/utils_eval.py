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