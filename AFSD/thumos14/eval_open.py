import argparse
import pickle
from AFSD.evaluation.eval_detection import ANETdetection
import os, json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('output_json', type=str)
parser.add_argument('gt_json', type=str, default='datasets/thumos14/annotations/thumos_gt.json', nargs='?')
parser.add_argument('--cls_idx_known', type=str)
parser.add_argument('--all_splits', nargs='+', type=int)
parser.add_argument('--open_set', action='store_true')
parser.add_argument('--trainset_result', type=str)
args = parser.parse_args()

tious = [0.3, 0.4, 0.5, 0.6, 0.7]


def read_threshold(trainset_result):
    assert os.path.exists(trainset_result), 'File does not exist! %s'%(trainset_result)
    with open(trainset_result, 'r') as fobj:
        data = json.load(fobj)
        threshold = data['external_data']['threshold']
    return threshold


mAPs_all, average_mAP_all = [], []
mWIs_all, average_mWI_all = [], []
for split in args.all_splits:
    # GT file and Pred file
    gt_file = args.gt_json if args.open_set else args.gt_json.format(id=split)
    pred_file = args.output_json.format(id=split)
    cls_idx_known = args.cls_idx_known.format(id=split)
    # read threshold value
    threshold = read_threshold(args.trainset_result.format(id=split)) if args.open_set else 0
    # instantiate evaluator
    anet_detection = ANETdetection(
        ground_truth_filename=gt_file,
        prediction_filename=pred_file,
        cls_idx_detection=cls_idx_known,
        subset='test', 
        openset=args.open_set,
        ood_threshold=threshold,
        tiou_thresholds=tious,
        verbose=False)
    print(f'Running the evaluation on split {split} with threshold {threshold:.12f}...')
    # run evaluation
    mAPs, average_mAP, ap = anet_detection.evaluate(type='AP')
    if args.open_set:
        mWIs, average_mWI, wi = anet_detection.evaluate(type='WI')
        with open(os.path.join(os.path.dirname(pred_file), 'open_stats.pkl'), 'wb') as f:
            pickle.dump(anet_detection.stats, f, pickle.HIGHEST_PROTOCOL)
    
    # report
    eval_filename = 'eval_open.txt' if args.open_set else 'eval.txt'
    with open(os.path.join(os.path.dirname(pred_file), eval_filename), 'w') as f:
        for (tiou, mAP) in zip(tious, mAPs):
            f.writelines(f"mAP at tIoU {tiou} is {mAP:.5f}\n")
        f.writelines(f"Average mAP is {average_mAP:.5f}")
        if args.open_set:
            f.writelines('\n')
            for (tiou, mWI) in zip(tious, mWIs):
                f.writelines(f"mWI at tIoU {tiou} is {mWI:.5f}\n")
            f.writelines(f"Average mWI is {average_mWI:.5f}")
    mAPs_all.append(mAPs)
    average_mAP_all.append(average_mAP)
    if args.open_set:
        mWIs_all.append(mWIs)
        average_mWI_all.append(average_mWI)

# print the averaged results
mAPs_mean = np.array(mAPs_all).mean(axis=0)
mAPs_std = np.array(mAPs_all).std(axis=0)
average_mAP_mean = np.array(average_mAP_all).mean(axis=0)
average_mAP_std = np.array(average_mAP_all).std(axis=0)
for (tiou, mean, std) in zip(tious, mAPs_mean, mAPs_std):
    print(f"mAP at tIoU {tiou} is {mean:.5f} ({std:.5f})")
print(f"Average mAP is {average_mAP_mean:.5f} ({average_mAP_std:.5f})\n")

if args.open_set:
    # print the averaged results
    mWIs_mean = np.array(mWIs_all).mean(axis=0)
    mWIs_std = np.array(mWIs_all).std(axis=0)
    average_mWI_mean = np.array(average_mWI_all).mean(axis=0)
    average_mWI_std = np.array(average_mWI_all).std(axis=0)
    for (tiou, mean, std) in zip(tious, mWIs_mean, mWIs_std):
        print(f"mWI at tIoU {tiou} is {mean:.5f} ({std:.5f})")
    print(f"Average mWI is {average_mWI_mean:.5f} ({average_mWI_std:.5f})\n")