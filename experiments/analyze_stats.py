import os, sys
import pickle
import matplotlib.pyplot as plt
import numpy as np


def get_mean_stds(data):
    return np.mean(data), np.std(data) / np.sqrt(len(data)) * 1.96

if __name__ == '__main__':
    split, exp_tag = sys.argv[1], sys.argv[2]
    tious = [0.3, 0.4, 0.5, 0.6, 0.7]

    stat_file = os.path.join(f'../output/{exp_tag}/split_{split}/open_stats.pkl')
    with open(stat_file, 'rb') as f:
        stats = pickle.load(f)

    Nums = np.zeros((len(tious), 7))
    width = 0.15
    fontsize = 18
    items = ['$TP_{u2u}$', '$TP_{k2k}$', '$FP_{u2k}$', '$FP_{k2k}$', '$FP_{k2u}$', '$FP_{bg2u}$', '$FP_{bg2k}$']
    colors = ['k', 'g', 'm', 'c', 'y']
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
    # plt.close()


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