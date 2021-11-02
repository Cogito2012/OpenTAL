import matplotlib.pyplot as plt
import os
import pickle


if __name__ == '__main__':

    labels = ['OpenTAL', 'EDL', 'SoftMax']
    result_folders = ['edl_oshead_iou', 'edl_15kc', 'default']
    split = '0'
    tiou_target = 0.3
    line_styles = ['r-', 'g-', 'b-']
    fontsize = 18
    fig_path = 'experiments/figs'
    os.makedirs(fig_path, exist_ok=True)

    # draw ROC Curve
    plt.figure(figsize=(6, 5))
    plt.rcParams["font.family"] = "Arial"
    for idx, (folder, label) in enumerate(zip(result_folders, labels)):
        # load result file
        result_file = os.path.join('output', folder, f'split_{split}', 'auc_data', 'roc_data.pkl')
        with open(result_file, 'rb') as f:
            roc_data = pickle.load(f)
        # draw curves
        for tidx, (fpr, tpr, auc, tiou) in enumerate(zip(roc_data['fpr'], roc_data['tpr'], roc_data['auc'], roc_data['tiou'])):
            if tiou == tiou_target:
                plt.plot(fpr, tpr, line_styles[idx], label=f'{label} ({auc*100:.2f})')
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'AUC_ROC_compare.png'))
    plt.savefig(os.path.join(fig_path, 'AUC_ROC_compare.pdf'))
    plt.close()

    # draw PR Curve
    plt.figure(figsize=(6, 5))
    plt.rcParams["font.family"] = "Arial"
    for idx, (folder, label) in enumerate(zip(result_folders, labels)):
        # load result file
        result_file = os.path.join('output', folder, f'split_{split}', 'auc_data', 'pr_data.pkl')
        with open(result_file, 'rb') as f:
            pr_data = pickle.load(f)
        # draw curves
        for tidx, (precision, recall, auc, tiou) in enumerate(zip(pr_data['precision'], pr_data['recall'], pr_data['auc'], pr_data['tiou'])):
            if tiou == tiou_target:
                plt.plot(recall, precision, line_styles[idx], label=f'{label} ({auc*100:.2f})')
    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'AUC_PR_compare.png'))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.rcParams["font.family"] = "Arial"
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, 'placeholder.png'))
    plt.savefig(os.path.join(fig_path, 'placeholder.pdf'))

