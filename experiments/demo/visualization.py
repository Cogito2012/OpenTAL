import os, json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    segments, labels, scores = [], [], []
    if len(idx) > 0:
        segments = [data['segments'][i] for i in idx]
        labels = [data['labels'][i] for i in idx]
        if type == 'pred_u':
            scores = [data['scores']['uncertainty'][i] for i in idx]
        elif type == 'pred_conf':
            scores = [1 - data['scores']['confidence'][i] for i in idx]
        segments = np.array(segments)
    return segments, labels, scores


def match_preds_with_gt(segment_preds, label_preds, scores_pred, segment_gts, thresh=0.25, tiou_threshold=0.3):
    actions_pred = {'segments': [], 'labels': [], 'matched_gtid': []}
    # match predictions with ground truth by tIoU
    lock_gt = np.ones((len(segment_gts))) * -1
    for idx, (seg, label, score) in enumerate(zip(segment_preds, label_preds, scores_pred)):
        tiou_arr = segment_iou(seg, segment_gts)
        tiou_sorted_idx = tiou_arr.argsort()[::-1]  # tIoU in a decreasing order
        for jdx in tiou_sorted_idx:
            if tiou_arr[jdx] < tiou_threshold:  # background segment
                break
        if lock_gt[jdx] >= 0:
            continue  # this gt was matched before
        # for positive localized actions
        label = 'Unknown' if score > thresh else label
        actions_pred['segments'].append(seg)
        actions_pred['labels'].append(label)
        actions_pred['matched_gtid'].append(jdx)
        lock_gt[jdx] = idx
    return actions_pred


def get_thresholds(predictions, ground_truths, video_list, method, unct_threshold_openmax, trainset_result, tiou=0.3):
    thresholds = {}
    if method == 'OpenTAL':
        # unct_threshold = unct_threshold_opental
        for video_name in tqdm(video_list, total=len(video_list), desc='Searching for thresholds: '):
            segment_gts, label_gts, _ = retrieve_segments(ground_truths, video_name, type='gt')
            segment_preds, label_preds, scores_pred = retrieve_segments(predictions, video_name, type='pred_u')
            # search for the best thresholds
            candidates = np.arange(0.05, 1.0, 0.05)
            all_cnts = np.zeros((len(candidates),))
            for i, t in enumerate(candidates):
                actions_pred = match_preds_with_gt(segment_preds, label_preds, scores_pred, segment_gts, \
                    thresh=t, tiou_threshold=tiou)
                cnt = 0
                for label_pred, jdx in zip(actions_pred['labels'], actions_pred['matched_gtid']):
                    if label_gts[jdx] == label_pred:
                        cnt = cnt + 1
                    else:
                        cnt = cnt - 1
                all_cnts[i] = cnt
            thresholds[video_name] = candidates[np.argmax(all_cnts)]
        print(thresholds)
    elif method == 'OpenMax':
        for video_name in video_list:
            thresholds[video_name] = unct_threshold_openmax
    else:
        thresh = read_threshold(trainset_result)
        for video_name in video_list:
            thresholds[video_name] = thresh
    return thresholds

 
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



def draw_action_detections(fig_file, all_actions, actions_gt, durations, cls_to_color, fontsize=18, paper_format=False):
    fig, axes = plt.subplots(1, 1, figsize=(15, 5), sharex=True)
    plt.rcParams["font.family"] = "Arial"
    video_height = 30
    video_len = 500  # pixels
    offset = 5
    line_color = (128, 138, 135)
    video_duration = durations[1] + 3  # seconds
    xlocs = np.arange(video_len+1, step=100)
    xlabels = ['%.1f'%(loc / video_len * video_duration) for loc in xlocs]


    fig_height = (video_height + offset + 2) * (len(all_actions) + 1)
    video_bars = np.ones((fig_height, video_len, 3), dtype=np.uint8) * 255
    # draw GT bar
    r_start = 0
    r_end = video_height
    for seg, label in zip(actions_gt['segments'], actions_gt['labels']):
        c_start = int(video_len / video_duration * seg[0])
        c_end = int(video_len / video_duration * seg[1])
        color = cls_to_color[label] if label in cls_to_color else (0, 0, 0)  # black: novel class
        video_bars[r_start: r_end, c_start: c_end + 1, :] = color
    # draw upper and bottom lines
    video_bars[r_start: r_start + 1, :, :] = line_color
    video_bars[r_end: r_end + 1, :, :] = line_color

    # draw Pred bars
    for i, (method, actions_pred) in enumerate(all_actions.items()):
        r_start += (video_height + offset)
        r_end += (video_height + offset)
        for seg, label in zip(actions_pred['segments'], actions_pred['labels']):
            c_start = int(video_len / video_duration * seg[0])
            c_end = int(video_len / video_duration * seg[1])
            color = cls_to_color[label] if label != 'Unknown' else (0, 0, 0)
            video_bars[r_start: r_end, c_start: c_end + 1, :] = color
        # draw upper and bottom lines
        video_bars[r_start: r_start + 1, :, :] = line_color
        video_bars[r_end: r_end + 1, :, :] = line_color
        
    # visualize
    axes.imshow(video_bars)
    axes.set_facecolor((1.0, 0.47, 0.42))
    axes.set_frame_on(False)
    axes.get_xaxis().tick_bottom()
    axes.get_yaxis().set_visible(False)
    xmin, xmax = axes.get_xaxis().get_view_interval()
    ymin, ymax = axes.get_yaxis().get_view_interval()
    axes.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    axes.set_yticks([])
    axes.set_xticks(xlocs)
    axes.set_xticklabels(xlabels, fontsize=fontsize)
    # draw labels
    r_center = int(video_height * 0.5)
    left_border = -80
    axes.text(left_border, r_center, 'Ground Truth', horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
    for i, method in enumerate(list(all_actions.keys())):
        r_center += (video_height + offset)
        axes.text(left_border, r_center, method, horizontalalignment='left', verticalalignment='center', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(fig_file)
    if paper_format:
        plt.savefig(fig_file[:-4] + '.pdf')
    plt.close()



def main():
    split = '0'
    exp_tags = ['softmax', 'openmax', 'open_edl', 'opental_final']
    method_list = ['SoftMax', 'OpenMax', 'EDL', 'OpenTAL']
    # selected_images = ['video_test_0000039', 'video_test_0000379', 'video_test_0001081', 'video_test_0001468', 'video_test_0001484']
    tiou_threshold = 0.3
    unct_threshold_opental = 0.25  # the best threshold
    unct_threshold_openmax = 0.995  # the best threshold
    random.seed(123)
    np.random.seed(123)

    # save path
    save_path = f'experiments/demo/vis_compare_all0'
    os.makedirs(save_path, exist_ok=True)

    # # annotation infos
    # video_info_path = './datasets/thumos14/annotations_open/test_video_info.csv'
    # video_data_path = './datasets/thumos14/test_npy/'

    # GT annotations
    gt_file = 'datasets/thumos14/annotations/thumos_gt.json'  # all videos (known + unknown)
    # import ground truth and parse into segmental level dictionary
    ground_truths, video_durations = import_ground_truth(gt_file)
    video_list = list(set(ground_truths['videos']))
    
    # import known actions
    known_class_file = f'datasets/thumos14/annotations_open/split_{split}/Class_Index_Known.txt'
    activity_index = get_activity_index(known_class_file)
    colors = ncolors(len(activity_index))
    cls_to_color = {}
    for cls, idx in activity_index.items():
        cls_to_color[cls] = colors[idx]

    all_predictions = {}
    for method, tag in zip(method_list, exp_tags):
        pred_file = f'output/{tag}/split_{split}/thumos14_open_rgb.json'
        # import predictions according the GT videos
        score_items = ['uncertainty'] if method in ['OpenTAL', 'EDL'] else ['confidence']
        predictions = import_prediction(pred_file, video_list, activity_index, score_items=score_items)
        # import the threshold from train set
        trainset_result = f'output/{tag}/split_{split}/thumos14_open_trainset.json'
        thresholds = get_thresholds(predictions, ground_truths, video_list, method, unct_threshold_openmax, trainset_result, tiou=tiou_threshold)
        all_predictions[method] = {'pred': predictions, 'threshold': thresholds}

    # draw all validation videos
    for video_name in tqdm(video_list, total=len(video_list), desc='Creating Demos'):
        # if video_name not in selected_images:
        #     continue
        # retrieve the ground truth of this video
        segment_gts, label_gts, _ = retrieve_segments(ground_truths, video_name, type='gt')
        actions_gt = {'segments': segment_gts, 'labels': label_gts}
        durations = video_durations[video_name]
 
        # retrieve the predictions of this video for each method
        all_actions = {}
        for method in method_list:
            type = 'pred_u' if method in ['OpenTAL', 'EDL'] else 'pred_conf'
            segment_preds, label_preds, scores_pred = retrieve_segments(all_predictions[method]['pred'], video_name, type=type)
            # match the predictions with GT
            actions_pred = match_preds_with_gt(segment_preds, label_preds, scores_pred, segment_gts, \
                thresh=all_predictions[method]['threshold'][video_name], tiou_threshold=tiou_threshold)
            all_actions[method] = actions_pred
        
        # draw figure
        fig_file = os.path.join(save_path, f'{video_name}.png')
        draw_action_detections(fig_file, all_actions, actions_gt, durations, cls_to_color, fontsize=22, paper_format=True)



if __name__ == '__main__':
    
    main()