import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def get_mean_stds(data):
    return np.mean(data), np.std(data) / np.sqrt(len(data)) * 1.96


if __name__ == '__main__':

    labels = ['OpenTAL', 'EDL', 'SoftMax']
    result_folders = ['edl_oshead_iou', 'edl_15kc', 'default']
    colors = ['k', 'g', 'm']
    split = '0'
    tiou_target = 0.3
    tidx = 0  # 0-4 for [0,3...,0.7]

    items = ['$TP_{u2u}$', '$TP_{k2k}$', '$FP_{u2k}$', '$FP_{k2k}$', '$FP_{k2u}$', '$FP_{bg2u}$', '$FP_{bg2k}$']
    fontsize = 18
    width = 0.25
    fig_path = 'experiments/figs'
    os.makedirs(fig_path, exist_ok=True)


    xrng = np.arange(len(items))
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    plt.rcParams["font.family"] = "Arial"
    for idx, (folder, label, color) in enumerate(zip(result_folders, labels, colors)):
        # load result file
        result_file = os.path.join('output', folder, f'split_{split}', 'open_stats.pkl')
        with open(result_file, 'rb') as f:
            stats = pickle.load(f)
        print(label)
        all_scores = 1 - np.array(stats['ood_scores'])
        mean_scores = np.zeros((7))
        std_scores = np.zeros((7))
        mean_scores[0], std_scores[0] = get_mean_stds(all_scores[stats['tp_u2u'][tidx] > 0])
        mean_scores[1], std_scores[1] = get_mean_stds(all_scores[stats['tp_k2k'][tidx].sum(axis=0) > 0])
        mean_scores[2], std_scores[2] = get_mean_stds(all_scores[stats['fp_u2k'][tidx].sum(axis=0) > 0])
        mean_scores[3], std_scores[3] = get_mean_stds(all_scores[stats['fp_k2k'][tidx].sum(axis=0) > 0])
        mean_scores[4], std_scores[4] = get_mean_stds(all_scores[stats['fp_k2u'][tidx] > 0])
        mean_scores[5], std_scores[5] = get_mean_stds(all_scores[stats['fp_bg2u'][tidx] > 0])
        mean_scores[6], std_scores[6] = get_mean_stds(all_scores[stats['fp_bg2k'][tidx].sum(axis=0) > 0])

        h = ax.bar(xrng + (idx-1) * width, mean_scores, yerr=std_scores, width=width, label=f'{label}', align='center', alpha=0.5, ecolor='black', color=color)

    ax.set_ylim(0, 1.2)
    ax.set_ylabel('OOD Scores', fontsize=fontsize)
    ax.set_xticks(xrng)
    ax.set_xticklabels(items, fontsize=fontsize-3)
    ax.legend(fontsize=fontsize, loc='upper center', ncol=3)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'OOD_Score_compare.png'))