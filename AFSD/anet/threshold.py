import os
import numpy as np
import json
from AFSD.common.anet_dataset import load_json
from AFSD.common.config import config
from test import get_basic_config, inference_thread

import multiprocessing as mp
import threading



def compute_threshold(result_dict, scoring='confidence'):
    all_scores = []
    for vid, proposal_list in result_dict.items():
        for prop in proposal_list:
            if scoring == 'uncertainty':
                ood_score = 1 - prop['uncertainty']
            elif scoring == 'confidence':
                ood_score = prop['score']
            elif scoring == 'uncertainty_actionness':
                ood_score = 1 - prop['uncertainty'] * prop['actionness']
            all_scores.append(ood_score)
    score_sorted = np.sort(all_scores)  # sort the confidence score in an increasing order
    N = len(all_scores)
    topK = N - int(N * 0.95)
    threshold = score_sorted[topK-1]
    return threshold


def thresholding(cfg, output_file, thread_num=1):
    processes = []
    lock = threading.Lock()
    
    train_cls_data = load_json('datasets/activitynet/result_tsn_train.json')
    videos_in_clsdata = ['v_' + name for name in list(train_cls_data['results'].keys())]
    videos_in_annodata = list(cfg.video_infos.keys())
    video_list = list(set(videos_in_clsdata) & set(videos_in_annodata))

    video_num = len(video_list)
    per_thread_video_num = video_num // thread_num
    result_dict = mp.Manager().dict()

    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        # inference_thread(lock, i, sub_video_list, train_cls_data, cfg)
        p = mp.Process(target=inference_thread, args=(lock, i, sub_video_list, train_cls_data, cfg, result_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # compute threshold value
    threshold = compute_threshold(result_dict, scoring=cfg.scoring)

    output_dict = {"version": "ActivityNet-v1.3", "results": dict(result_dict), "external_data": {'threshold': threshold}}
    with open(output_file, "w") as out:
        json.dump(output_dict, out)
    return threshold


def main():
    cfg = get_basic_config(config, dataset='training')

    output_file = os.path.join(cfg.output_path, cfg.json_name)
    if not os.path.exists(output_file):
        threshold = thresholding(cfg, output_file, thread_num=cfg.thread_num)
    else:
        with open(output_file, 'r') as fobj:
            data = json.load(fobj)
            threshold = data['external_data']['threshold']
        print(f'Thresholding result file already exist at {output_file}!')

    print(f'The threshold is: {threshold:.12f}')


if __name__ == '__main__':
    # keep all things private in this file
    main()