import matplotlib.pyplot as plt
import os
import pickle


def draw_OSDR_curve(split, tiou_thresh, fig_name):
    plt.figure(figsize=(6, 5))
    plt.rcParams["font.family"] = "Arial"
    for idx, (folder, label) in enumerate(zip(result_folders, labels)):
        # load result file
        result_file = os.path.join('output', folder, f'split_{split}', 'auc_data', 'osdr_data.pkl')
        with open(result_file, 'rb') as f:
            osdr_data = pickle.load(f)
        # draw curves
        for tidx, (fpr, cdr, osdr, tiou) in enumerate(zip(osdr_data['fpr'], osdr_data['cdr'], osdr_data['osdr'], osdr_data['tiou'])):
            if tiou == tiou_thresh:
                plt.plot(fpr[:-2], cdr[:-2], line_styles[idx], linewidth=2, label=f'{label} ({osdr*100:.2f})')
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('Correct Detection Rate', fontsize=fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 0.7)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, fig_name + '.png'))
    plt.savefig(os.path.join(fig_path, fig_name + '.pdf'))
    plt.close()


def draw_ROC_curve(split, tiou_thresh, fig_name):
    plt.figure(figsize=(6, 5))
    plt.rcParams["font.family"] = "Arial"
    for idx, (folder, label) in enumerate(zip(result_folders, labels)):
        # load result file
        result_file = os.path.join('output', folder, f'split_{split}', 'auc_data', 'roc_data.pkl')
        with open(result_file, 'rb') as f:
            roc_data = pickle.load(f)
        # draw curves
        for tidx, (fpr, tpr, auc, tiou) in enumerate(zip(roc_data['fpr'], roc_data['tpr'], roc_data['auc'], roc_data['tiou'])):
            if tiou == tiou_thresh:
                plt.plot(fpr, tpr, line_styles[idx], linewidth=2, label=f'{label} ({auc*100:.2f})')
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, fig_name + '.png'))
    plt.savefig(os.path.join(fig_path, fig_name + '.pdf'))
    plt.close()


def draw_PR_curve(split, tiou_thresh, fig_name):
    plt.figure(figsize=(6, 5))
    plt.rcParams["font.family"] = "Arial"
    for idx, (folder, label) in enumerate(zip(result_folders, labels)):
        # load result file
        result_file = os.path.join('output', folder, f'split_{split}', 'auc_data', 'pr_data.pkl')
        with open(result_file, 'rb') as f:
            pr_data = pickle.load(f)
        # draw curves
        for tidx, (precision, recall, auc, tiou) in enumerate(zip(pr_data['precision'], pr_data['recall'], pr_data['auc'], pr_data['tiou'])):
            if tiou == tiou_thresh:
                plt.plot(recall, precision, line_styles[idx], label=f'{label} ({auc*100:.2f})')
    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, fig_name + '.png'))
    plt.savefig(os.path.join(fig_path, fig_name + '.pdf'))
    plt.close()


if __name__ == '__main__':

    labels = ['OpenTAL', 'EDL', 'OpenMax', 'SoftMax']
    result_folders = ['opental_final', 'open_edl', 'openmax', 'softmax']
    all_splits = ['0', '1', '2']
    tiou_targets = [0.3, 0.4, 0.5, 0.6, 0.7]
    line_styles = ['r-', 'g-', 'c-', 'b-']
    fontsize = 22
    fig_path = 'experiments/figs'
    os.makedirs(fig_path, exist_ok=True)

    # draw ROC Curve
    for split in all_splits:
        for tiou in tiou_targets:
            fig_name = f'AUC_ROC_split{split}_tiou{tiou}'
            draw_ROC_curve(split, tiou, fig_name)

    # draw PR Curve
    for split in all_splits:
        for tiou in tiou_targets:
            fig_name = f'AUC_PR_split{split}_tiou{tiou}'
            draw_PR_curve(split, tiou, fig_name)

    # draw OSDR Curve
    for split in all_splits:
        for tiou in tiou_targets:
            fig_name = f'OSDR_split{split}_tiou{tiou}'
            draw_OSDR_curve(split, tiou, fig_name)

