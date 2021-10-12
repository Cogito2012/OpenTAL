import os, sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
from AFSD.evaluation.utils_eval import interpolated_prec_rec


def get_activity_index(class_info_path, openset=False):
    txt = np.loadtxt(class_info_path, dtype=str)
    class_to_idx = {}
    if openset:
        class_to_idx['__unknown__'] = 0  # 0 is reserved for unknown in open set
    for idx, l in enumerate(txt):
        class_to_idx[l[1]] = idx + 1  # starting from 1 to K (K=15 for thumos14)
    return class_to_idx


def get_curve_points(prec, rec, interpolate=True):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    if not interpolate:
        return rec, prec
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    # ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return mrec[idx], mprec[idx]


def plot_stats(width=0.15, fontsize=18):
    Nums = np.zeros((len(tious), 7))
    xrng = np.arange(len(items)) 
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    for i, (iou, c) in enumerate(zip(tious, colors)):
        Nums[i, 0] = stats['tp_u2u'][i].sum()  # N_tp_u2u
        Nums[i, 1] = stats['tp_k2k'][i].sum()  # N_tp_k2k
        Nums[i, 2] = stats['fp_u2k'][i].sum()  # N_fp_u2k
        Nums[i, 3] = stats['fp_k2k'][i].sum()  # N_fp_k2k
        Nums[i, 4] = stats['fp_k2u'][i].sum()  # N_fp_k2u
        Nums[i, 5] = stats['fp_bg2u'][i].sum()  # N_fp_bg2u
        Nums[i, 6] = stats['fp_bg2k'][i].sum()  # N_fp_bg2k

        h = ax.bar(xrng + (i-2) * width, Nums[i], width, label=f'tIoU={iou}', color=c)
        if i == len(tious) - 1:
            ax.bar_label(h, padding=3)
    
    ax.set_ylabel('Number of Segments', fontsize=fontsize)
    ax.set_xticks(xrng)
    ax.set_xticklabels(items, fontsize=fontsize-3)
    ax.legend(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f'../output/{exp_tag}/split_{split}/stats.png')


def get_mean_stds(data):
    return np.mean(data), np.std(data) / np.sqrt(len(data)) * 1.96


def plot_scores(width=0.15, fontsize=18):
    xrng = np.arange(len(items)) 
    # score distribution
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    all_scores = np.array(stats['scores'])
    all_max_tious = stats['max_tious']
    mean_scores = np.zeros((len(tious), 7))
    std_scores = np.zeros((len(tious), 7))
    for i, (iou, c) in enumerate(zip(tious, colors)):
        mean_scores[i, 0], std_scores[i, 0] = get_mean_stds(all_scores[stats['tp_u2u'][i] > 0])
        mean_scores[i, 1], std_scores[i, 1] = get_mean_stds(all_scores[stats['tp_k2k'][i].sum(axis=0) > 0])
        mean_scores[i, 2], std_scores[i, 2] = get_mean_stds(all_scores[stats['fp_u2k'][i].sum(axis=0) > 0])
        mean_scores[i, 3], std_scores[i, 3] = get_mean_stds(all_scores[stats['fp_k2k'][i].sum(axis=0) > 0])
        mean_scores[i, 4], std_scores[i, 4] = get_mean_stds(all_scores[stats['fp_k2u'][i] > 0])
        mean_scores[i, 5], std_scores[i, 5] = get_mean_stds(all_scores[stats['fp_bg2u'][i] > 0])
        mean_scores[i, 6], std_scores[i, 6] = get_mean_stds(all_scores[stats['fp_bg2k'][i].sum(axis=0) > 0])

        h = ax.bar(xrng + (i-2) * width, mean_scores[i], yerr=std_scores[i], width=width, label=f'tIoU={iou}', align='center', alpha=0.5, ecolor='black', color=c)
        
    ax.set_ylabel('Confidence Scores of Segments', fontsize=fontsize)
    ax.set_xticks(xrng)
    ax.set_xticklabels(items, fontsize=fontsize-3)
    ax.legend(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f'../output/{exp_tag}/split_{split}/stats_scores.png')


def plot_max_tious(width=0.15, fontsize=18):
    xrng = np.arange(len(items)) 
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    all_max_tious = np.array(stats['max_tious'])

    mean_ious = np.zeros((len(tious), 7))
    std_ious = np.zeros((len(tious), 7))
    for i, (iou, c) in enumerate(zip(tious, colors)):
        mean_ious[i, 0], std_ious[i, 0] = get_mean_stds(all_max_tious[stats['tp_u2u'][i] > 0])
        mean_ious[i, 1], std_ious[i, 1] = get_mean_stds(all_max_tious[stats['tp_k2k'][i].sum(axis=0) > 0])
        mean_ious[i, 2], std_ious[i, 2] = get_mean_stds(all_max_tious[stats['fp_u2k'][i].sum(axis=0) > 0])
        mean_ious[i, 3], std_ious[i, 3] = get_mean_stds(all_max_tious[stats['fp_k2k'][i].sum(axis=0) > 0])
        mean_ious[i, 4], std_ious[i, 4] = get_mean_stds(all_max_tious[stats['fp_k2u'][i] > 0])
        mean_ious[i, 5], std_ious[i, 5] = get_mean_stds(all_max_tious[stats['fp_bg2u'][i] > 0])
        mean_ious[i, 6], std_ious[i, 6] = get_mean_stds(all_max_tious[stats['fp_bg2k'][i].sum(axis=0) > 0])

        h = ax.bar(xrng + (i-2) * width, mean_ious[i], yerr=std_ious[i], width=width, label=f'tIoU={iou}', align='center', alpha=0.5, ecolor='black', color=c)
        
    ax.set_ylabel('Max tIoU values', fontsize=fontsize)
    ax.set_xticks(xrng)
    ax.set_xticklabels(items, fontsize=fontsize-3)
    ax.legend(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f'../output/{exp_tag}/split_{split}/stats_tiou.png')


def plot_wi_curves(fontsize=18):

    result_curve_path = f'../output/{exp_tag}/split_{split}/wi_curves'
    os.makedirs(result_curve_path, exist_ok=True)

    tp_u2u, tp_k2k, fp_u2k, fp_k2k, fp_k2u, fp_bg2u, fp_bg2k, num_gt = \
        stats['tp_u2u'], stats['tp_k2k'], stats['fp_u2k'], stats['fp_k2k'], stats['fp_k2u'], stats['fp_bg2u'], stats['fp_bg2k'], stats['num_gt']
    fp_k2u += fp_bg2u
    fp_k2k += fp_bg2k

    # impact on recall ratio
    tp_u2u_cumsum = np.cumsum(tp_u2u, axis=-1).astype(float)  # T x N
    recall_ratio_cumsum = num_gt[1:].sum() / ( num_gt[1:].sum() + num_gt[0] - tp_u2u_cumsum)  # T x N
    # impact on precision ratio
    tp_k2k_cumsum = np.cumsum(tp_k2k, axis=-1).astype(float)  # T x K x N
    fp_u2k_cumsum = np.cumsum(fp_u2k, axis=-1).astype(float)  # T x K x N
    fp_k2k_cumsum = np.cumsum(fp_k2k, axis=-1).astype(float)  # T x K x N
    precision_ratio_cumsum = (tp_k2k_cumsum + fp_k2k_cumsum) / (tp_k2k_cumsum + fp_k2k_cumsum + fp_u2k_cumsum + 1e-6)

    known_classes = copy.deepcopy(activity_index)
    del known_classes['__unknown__']

    for clsname, cidx in known_classes.items():
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        for tidx, (iou, c) in enumerate(zip(tious, colors)):
            pi = precision_ratio_cumsum[tidx, cidx-1, :]
            ri = recall_ratio_cumsum[tidx, :]
            wi = interpolated_prec_rec(pi, ri)
            xpts, ypts = get_curve_points(pi, ri, interpolate=False)
            ax.plot(xpts, ypts, '-', color=c, label=f'tIoU={iou}')
        ax.legend(fontsize=fontsize, loc='lower right')
        ax.set_xlim(0.7, 0.8)
        ax.set_ylim(0, 1)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(f'{clsname} (AWI = {wi * 100:.2f}%)', fontsize=fontsize)
        plt.xlabel('Recall Impact', fontsize=fontsize)
        plt.ylabel('Precision Impact', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(result_curve_path, f'wi_{clsname}.png'))
        plt.close()


if __name__ == '__main__':
    exp_tag = sys.argv[1]
    split = '0'
    tious = [0.3, 0.4, 0.5, 0.6, 0.7]

    stat_file = os.path.join(f'../output/{exp_tag}/split_{split}/open_stats.pkl')
    with open(stat_file, 'rb') as f:
        stats = pickle.load(f)

    items = ['$TP_{u2u}$', '$TP_{k2k}$', '$FP_{u2k}$', '$FP_{k2k}$', '$FP_{k2u}$', '$FP_{bg2u}$', '$FP_{bg2k}$']
    colors = ['k', 'g', 'm', 'c', 'y']  # for 5 tIoU thresholds

    activity_index = get_activity_index(f'../datasets/thumos14/annotations_open/split_{split}/Class_Index_Known.txt', openset=True)
    
    plot_stats()

    plot_scores()

    plot_max_tious()

    plot_wi_curves()

