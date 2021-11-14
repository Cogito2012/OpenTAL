import argparse
import numpy as np
from AFSD.evaluation.eval_detection import ANETdetection
import os, json, pickle

parser = argparse.ArgumentParser()
parser.add_argument('output_json', type=str)
parser.add_argument('gt_json', type=str, nargs='?')
parser.add_argument('--cls_idx_known', type=str)
parser.add_argument('--all_splits', nargs='+', type=int)
parser.add_argument('--open_set', action='store_true')
parser.add_argument('--draw_auc', action='store_true')
parser.add_argument('--ood_scoring', type=str, default='confidence', choices=['uncertainty', 'confidence', 'uncertainty_actionness'])
args = parser.parse_args()

tious = [0.1, 0.2, 0.3, 0.4, 0.5]


def write_eval_open(eval_file, tious, auc_ROC, auc_PR, OSDR):
    with open(eval_file, 'w') as f:
        for (tiou, auc_roc, auc_pr, osdr) in zip(tious, auc_ROC, auc_PR, OSDR):
            f.writelines(f"tIoU={tiou}: auc_roc={auc_roc:.5f}, auc_pr={auc_pr:.5f}, osdr={osdr:.5f}\n")
        f.writelines(f"Average AUC_ROC: {auc_ROC.mean():.5f}, Average AUC_PR: {auc_PR.mean():.5f}, Average OSDR: {OSDR.mean():.5f}\n")


def write_eval_closed(eval_file, tious, mAPs):
    with open(eval_file, 'w') as f:
        for (tiou, mAP) in zip(tious, mAPs):
            f.writelines(f"tIoU={tiou}: mAP={mAP:.5f}\n")
        f.writelines(f"Average mAP: {average_mAP:.5f}\n")


mAPs_all, average_mAP_all = [], []
aucROCs_all, average_aucROC_all = [], []
aucPRs_all, average_aucPR_all = [], []
OSDR_all, average_OSDR_all = [], []
for split in args.all_splits:
    # GT file and Pred file
    gt_file = args.gt_json.format(id=split)
    pred_file = args.output_json.format(id=split)
    cls_idx_known = args.cls_idx_known.format(id=split)
    auc_data_path = None
    if args.draw_auc:
        auc_data_path = os.path.join(os.path.join(os.path.dirname(pred_file), 'auc_data'))
        os.makedirs(auc_data_path, exist_ok=True)
    # instantiate evaluator
    anet_detection = ANETdetection(
        ground_truth_filename=gt_file,
        prediction_filename=pred_file,
        cls_idx_detection=cls_idx_known,
        subset=['validation'], 
        openset=args.open_set,
        ood_scoring=args.ood_scoring,
        tiou_thresholds=tious,
        draw_auc=args.draw_auc,
        curve_data_path=auc_data_path,
        verbose=False,
        dataset='anet')
    
    # run evaluation
    if args.open_set:
        anet_detection.pre_evaluate()
        # evaluate AUC of ROC and PR
        auc_ROC, auc_PR = anet_detection.evaluate(type='AUC')
        # evaluate auc of OSDR
        OSDR = anet_detection.evaluate(type='OSDR')
        aucROCs_all.append(auc_ROC)
        average_aucROC_all.append(auc_ROC.mean())
        aucPRs_all.append(auc_PR)
        average_aucPR_all.append(auc_PR.mean())
        OSDR_all.append(OSDR)
        average_OSDR_all.append(OSDR.mean())
        # output
        eval_file = os.path.join(os.path.dirname(pred_file), 'eval_open.txt')
        write_eval_open(eval_file, tious, auc_ROC, auc_PR, OSDR)
    else:
        mAPs, average_mAP, ap = anet_detection.evaluate(type='AP')
        mAPs_all.append(mAPs)
        average_mAP_all.append(average_mAP)
        # output
        eval_file = os.path.join(os.path.dirname(pred_file), 'eval.txt')
        write_eval_closed(eval_file, tious, mAPs)


def get_mean_std(data, axis=0):
    mean = np.array(data).mean(axis=axis)
    # see the Confidence Intervals: http://www.stat.yale.edu/Courses/1997-98/101/confint.htm
    std = np.array(data).std(axis=axis) / np.sqrt(len(data)) * 1.96
    return mean, std

if args.open_set:
    # print the averaged results of auc_ROC
    aucROCs_mean, aucROCs_std = get_mean_std(aucROCs_all, axis=0)
    average_aucROC_mean, average_aucROC_std = get_mean_std(average_aucROC_all, axis=0)
    # print the averaged results of auc_ROC
    aucPRs_mean, aucPRs_std = get_mean_std(aucPRs_all, axis=0)
    average_aucPR_mean, average_aucPR_std = get_mean_std(average_aucPR_all, axis=0)
    # print the averaged results of OSDR
    osdr_mean, osdr_std = get_mean_std(OSDR_all, axis=0)
    average_OSDR_mean, average_OSDR_std = get_mean_std(average_OSDR_all, axis=0)

    for (tiou, mean, std) in zip(tious, aucROCs_mean, aucROCs_std):
        print(f"AUC_ROC(tIoU={tiou}): mean={mean:.5f}, std={std:.5f}")
    print(f"Average AUC_ROC = {average_aucROC_mean:.5f} ({average_aucROC_std:.5f})\n")

    for (tiou, mean, std) in zip(tious, aucPRs_mean, aucPRs_std):
        print(f"AUC_PR(tIoU={tiou}): mean={mean:.5f}, std={std:.5f}")
    print(f"Average AUC_PR = {average_aucPR_mean:.5f} ({average_aucPR_std:.5f})\n")

    for (tiou, mean, std) in zip(tious, osdr_mean, osdr_std):
        print(f"OSDR(tIoU={tiou}): mean={mean:.5f}, std={std:.5f}")
    print(f"Average OSDR = {average_OSDR_mean:.5f} ({average_OSDR_std:.5f})\n")
else:
    # print the averaged results
    mAPs_mean, mAPs_std = get_mean_std(mAPs_all, axis=0)
    average_mAP_mean, average_mAP_std = get_mean_std(average_mAP_all, axis=0)
    for (tiou, mean, std) in zip(tious, mAPs_mean, mAPs_std):
        print(f"mAP(tIoU={tiou}): mean={mean:.5f}, std={std:.5f}")
    print(f"Average mAP is {average_mAP_mean:.5f} ({average_mAP_std:.5f})\n")