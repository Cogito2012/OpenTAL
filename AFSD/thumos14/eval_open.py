import argparse
from AFSD.evaluation.eval_detection import ANETdetection
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('output_json', type=str)
parser.add_argument('gt_json', type=str, default='datasets/thumos14/annotations/thumos_gt.json', nargs='?')
parser.add_argument('--all_splits', nargs='+', type=int)
parser.add_argument('--open_set', action='store_true')
args = parser.parse_args()

tious = [0.3, 0.4, 0.5, 0.6, 0.7]

mAPs_all, average_mAP_all = [], []
for split in args.all_splits:
    print(f'Running the evaluation on split {split}...')
    # GT file and Pred file
    gt_file = args.gt_json if args.open_set else args.gt_json.format(id=split)
    pred_file = args.output_json.format(id=split)
    # instantiate evaluator
    anet_detection = ANETdetection(
        ground_truth_filename=gt_file,
        prediction_filename=pred_file,
        subset='test', 
        tiou_thresholds=tious,
        verbose=False)
    # run evaluation
    mAPs, average_mAP, ap = anet_detection.evaluate()
    # report
    eval_filename = 'eval_open.txt' if args.open_set else 'eval.txt'
    with open(os.path.join(os.path.dirname(pred_file), eval_filename), 'w') as f:
        for (tiou, mAP) in zip(tious, mAPs):
            f.writelines(f"mAP at tIoU {tiou} is {mAP:.5f}\n")
        f.writelines(f"Average mAP is {average_mAP:.5f}")
    mAPs_all.append(mAPs)
    average_mAP_all.append(average_mAP)

# print the averaged results
mAPs_mean = np.array(mAPs_all).mean(axis=0)
mAPs_std = np.array(mAPs_all).std(axis=0)
average_mAP_mean = np.array(average_mAP_all).mean(axis=0)
average_mAP_std = np.array(average_mAP_all).std(axis=0)
for (tiou, mean, std) in zip(tious, mAPs_mean, mAPs_std):
    print(f"mAP at tIoU {tiou} is {mean:.5f} ({std:.5f})")
print(f"Average mAP is {average_mAP_mean:.5f} ({average_mAP_std:.5f})")