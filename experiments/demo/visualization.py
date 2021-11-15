import os, json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import colorsys
import random
from AFSD.evaluation.utils_eval import segment_iou


def read_threshold(trainset_result, type='confidence'):
    assert os.path.exists(trainset_result), 'File does not exist! %s'%(trainset_result)
    with open(trainset_result, 'r') as fobj:
        data = json.load(fobj)
        threshold = data['external_data']['threshold']
    if type == 'confidence':
        threshold = 1 - threshold
    return threshold


def get_activity_index(class_info_path):
    txt = np.loadtxt(class_info_path, dtype=str)
    class_to_idx = {'Unknown': 0}
    for idx, l in enumerate(txt):
        class_to_idx[l[1]] = idx + 1   # starting from 1 to K
    return class_to_idx


def import_ground_truth(gt_file, subset='test'):
    with open(gt_file, 'r') as fobj:
        data = json.load(fobj)

    videos, segments, labels = [], [], []
    durations = {}
    for videoid, v in data['database'].items():
        if v['subset'] != subset:
            continue
        min_start, max_end = np.inf, -np.inf
        for ann in v['annotations']:
            videos.append(videoid)
            start = float(ann['segment'][0])
            end = float(ann['segment'][1])
            segments.append([start, end])
            labels.append(ann['label'])
            # find the min start and max end
            min_start = start if start < min_start else min_start
            max_end = end if end > max_end else max_end
        durations[videoid] = [min_start, max_end]
    segments = np.array(segments)
    ground_truth = {'videos': videos, 'segments': segments, 'labels': labels}
    return ground_truth, durations


def import_prediction(pred_file, video_list, activity_index, score_items=['uncertainty', 'actionness']):
    with open(pred_file, 'r') as fobj:
        data = json.load(fobj)
    videos, segments, labels = [], [], []
    scores = {}
    for item in score_items:
        scores[item] = []
    for videoid, v in data['results'].items():
        if videoid not in video_list:
            continue
        for result in v:
            if result['label'] not in activity_index:
                continue
            # load videos
            videos.append(videoid)
            # load segments
            start = float(result['segment'][0])
            end = float(result['segment'][1])
            segments.append([start, end])
            # load labels
            labels.append(result['label'])
            # load scores
            if 'uncertainty' in score_items:
                scores['uncertainty'].append(result['uncertainty'])
            if 'confidence' in score_items:
                scores['confidence'].append(result['score'])
            if 'actionness' in score_items:
                scores['actionness'].append(result['actionness'])
    predictions = {'videos': videos, 'segments': segments, 'labels': labels, 'scores': scores}
    return predictions


def retrieve_segments(data, video_name, type='gt'):
    idx = [i for i,x in enumerate(data['videos']) if x == video_name]
    if type == 'gt':
        segments, labels = [], []
        if len(idx) > 0:
            segments = [data['segments'][i] for i in idx]
            labels = [data['labels'][i] for i in idx]
        segments = np.array(segments)
        return segments, labels
    
    if type == 'pred_ua':
        segments, labels, uncertainty, actionness = [], [], [], []  
        if len(idx) > 0: 
            segments = [data['segments'][i] for i in idx]
            labels = [data['labels'][i] for i in idx]
            uncertainty = [data['scores']['uncertainty'][i] for i in idx]
            actionness = [data['scores']['actionness'][i] for i in idx]
        segments = np.array(segments)
        return segments, labels, uncertainty, actionness
    
    if type == 'pred_conf':
        segments, labels, confs = [], [], []
        if len(idx) > 0: 
            segments = [data['segments'][i] for i in idx]
            labels = [data['labels'][i] for i in idx]
            confs = [data['scores']['confidence'][i] for i in idx]
        segments = np.array(segments)
        return segments, labels, confs


def clear_overlapping(actions):
    # segments, labels, ious = np.array(actions['segments']), actions['labels'], actions['ious']
    # for seg in segments:
    #     self_ious = segment_iou(seg, segments)
    return actions
 
def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
    return hls_colors
 
def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
 
    return rgb_colors



def draw_action_detections(fig_file, actions_pred, actions_gt, durations, cls_to_color, fontsize=18):
    fig, axes = plt.subplots(2, 1, figsize=(15, 4), sharex=True)
    video_len = 500  # pixels
    video_duration = durations[1] + 3  # seconds
    xlocs = np.arange(video_len+1, step=100)
    xlabels = ['%.2f'%(loc / video_len * video_duration) for loc in xlocs]
    # plot pred
    video_bar = np.ones((30, video_len, 3), dtype=np.uint8) * 255
    # texts = {'x': [], 'y': [], 'label': []}
    for seg, label in zip(actions_pred['segments'], actions_pred['labels']):
        start = int(video_len / video_duration * seg[0])
        end = int(video_len / video_duration * seg[1])
        color = cls_to_color[label] if label != 'Unknown' else (0, 0, 0)
        video_bar[:, start: end + 1, :] = color
        # texts['x'].append(int(video_len / video_duration * (seg[0] + 0.5)))
        # texts['y'].append(25)
        # texts['label'].append(label)
    axes[0].imshow(video_bar)
    axes[0].set_facecolor((1.0, 0.47, 0.42))
    # axes[0].get_yaxis().set_visible(False)
    axes[0].set_yticks([])
    axes[0].set_xticks(xlocs)
    axes[0].set_xticklabels(xlabels, fontsize=fontsize)
    axes[0].set_ylabel('Prediction', fontsize=fontsize)
    # # add labels
    # for x, y, str in zip(texts['x'], texts['y'], texts['label']):
    #     axes[0].text(x, y, str, fontsize=fontsize)

    # plot gt
    video_bar = np.ones((30, video_len, 3), dtype=np.uint8) * 255
    for seg, label in zip(actions_gt['segments'], actions_gt['labels']):
        start = int(video_len / video_duration * seg[0])
        end = int(video_len / video_duration * seg[1])
        color = cls_to_color[label] if label in cls_to_color else (0, 0, 0)  # black: novel class
        video_bar[:, start: end + 1, :] = color
    axes[1].imshow(video_bar)
    axes[1].set_facecolor((1.0, 0.47, 0.42))
    # axes[1].get_yaxis().set_visible(False)
    axes[1].set_yticks([])
    axes[1].set_xticks(xlocs)
    axes[1].set_xticklabels(xlabels, fontsize=fontsize)
    axes[1].set_ylabel('Ground Truth', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(fig_file)
    plt.close()



if __name__ == '__main__':
    
    split = '0'
    exp_tag = 'opental_final'
    tiou_threshold = 0.3
    random.seed(123)
    np.random.seed(123)

    # prediction results
    pred_file = f'output/{exp_tag}/split_{split}/thumos14_open_rgb.json'
    gt_file = 'datasets/thumos14/annotations/thumos_gt.json'  # all videos (known + unknown)
    known_class_file = f'datasets/thumos14/annotations_open/split_{split}/Class_Index_Known.txt'
    # save path
    save_path = f'output/{exp_tag}/split_{split}/demos'
    os.makedirs(save_path, exist_ok=True)

    # # annotation infos
    # video_info_path = './datasets/thumos14/annotations_open/test_video_info.csv'
    # video_data_path = './datasets/thumos14/test_npy/'

    # trainset_result = f'output/{exp_tag}/split_{split}/thumos14_open_trainset.json'
    # unct_threshold = read_threshold(trainset_result)
    unct_threshold = 0.25  # the best threshold

    # import known actions
    activity_index = get_activity_index(known_class_file)
    colors = ncolors(len(activity_index))
    cls_to_color = {}
    for cls, idx in activity_index.items():
        cls_to_color[cls] = colors[idx]

    # import ground truth and parse into segmental level dictionary
    ground_truths, video_durations = import_ground_truth(gt_file)
    video_list = list(set(ground_truths['videos']))

    # import predictions according the GT videos
    predictions = import_prediction(pred_file, video_list, activity_index, score_items=['uncertainty', 'actionness'])


    # draw all validation videos
    for video_name in tqdm(video_list, total=len(video_list), desc='Creating Demos'):
        # retrieve the predictions and ground truth of this video
        segment_preds, label_preds, unct_preds, act_preds = retrieve_segments(predictions, video_name, type='pred_ua')
        segment_gts, label_gts = retrieve_segments(ground_truths, video_name, type='gt')
        durations = video_durations[video_name]
        # # find the video with more than 1 class
        # if len(list(set(label_gts))) == 1:
        #     continue

        actions_pred = {'segments': [], 'labels': [], 'uncts': [], 'acts': []}
        ious_matched = []
        # match predictions with ground truth by tIoU
        lock_gt = np.ones((len(segment_gts))) * -1
        for idx, (seg, label, unct, act) in enumerate(zip(segment_preds, label_preds, unct_preds, act_preds)):
            tiou_arr = segment_iou(seg, segment_gts)
            tiou_sorted_idx = tiou_arr.argsort()[::-1]  # tIoU in a decreasing order
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_threshold:  # background segment
                    break
            if lock_gt[jdx] >= 0:
                continue  # this gt was matched before
            # for positive localized actions
            label = 'Unknown' if unct > unct_threshold else label

            actions_pred['segments'].append(seg)
            actions_pred['labels'].append(label)
            ious_matched.append(tiou_arr[jdx])
            lock_gt[jdx] = idx
        # clean the detected overlapping actions 
        actions_pred = clear_overlapping(actions_pred)
        actions_gt = {'segments': segment_gts, 'labels': label_gts}
        # draw figure
        fig_file = os.path.join(save_path, f'{video_name}.png')
        draw_action_detections(fig_file, actions_pred, actions_gt, durations, cls_to_color)
