from re import search
from matplotlib import colors
import torch
import torch.nn as nn
import os, sys
import numpy as np
import tqdm
import json
from AFSD.common import videotransforms
from AFSD.common.thumos_dataset import get_video_info, get_video_anno, get_class_index_map, annos_transform
from AFSD.thumos14.BDNet import BDNet, DirichletLayer
from test import get_offsets, prepare_data, prepare_clip, parse_output, decode_predictions, filtering, get_video_detections, get_path
from AFSD.common.config import config
from AFSD.evaluation.eval_detection import ANETdetection
import matplotlib.pyplot as plt
import pandas as pd
from AFSD.evaluation.utils_eval import segment_iou


def get_basic_config(config):
    class cfg: pass
    cfg.num_classes = config['dataset']['num_classes']
    cfg.conf_thresh = config['testing']['conf_thresh']
    cfg.top_k = config['testing']['top_k']
    cfg.nms_thresh = config['testing']['nms_thresh']
    cfg.nms_sigma = config['testing']['nms_sigma']
    cfg.clip_length = config['dataset']['testing']['clip_length']
    cfg.stride = config['dataset']['testing']['clip_stride']
    cfg.use_edl = config['model']['use_edl'] if 'use_edl' in config['model'] else False
    if cfg.use_edl:
        cfg.evidence = config['model']['evidence']    
    cfg.os_head = config['model']['os_head'] if 'os_head' in config['model'] else False
    if cfg.os_head:
        cfg.num_classes = cfg.num_classes - 1
    cfg.output_path = config['testing']['output_path']
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path)
    cfg.fusion = config['testing']['fusion']
    cfg.scoring = config['testing']['ood_scoring']
    cfg.cls_idx_known = config['dataset']['class_info_path']
    cfg.open_set = config['open_set']
    cfg.split = config['testing']['split']
    cfg.overlap_thresh = config['training']['piou']

    cfg.rgb_data_path, cfg.flow_data_path = {}, {}
    ###  specific for training set
    cfg.rgb_data_path['training'] = config['training'].get('rgb_data_path',
                                        './datasets/thumos14/validation_npy/')
    cfg.flow_data_path['training'] = config['training'].get('flow_data_path',
                                        './datasets/thumos14/validation_flow_npy/')
    ###  specific for testing set
    cfg.rgb_data_path['testing'] = config['testing'].get('rgb_data_path',
                                        './datasets/thumos14/test_npy/')
    cfg.flow_data_path['testing'] = config['testing'].get('flow_data_path',
                                        './datasets/thumos14/test_flow_npy/')
    cfg.gt_known_json = 'datasets/thumos14/annotations_open/split_{id:d}/known_gt.json'
    cfg.gt_all_json  ='datasets/thumos14/annotations/thumos_gt.json'
    return cfg



def build_model(fusion=False):
    net, flow_net = None, None
    if fusion:
        rgb_net = BDNet(in_channels=3, training=False, use_edl=cfg.use_edl)
        flow_net = BDNet(in_channels=2, training=False, use_edl=cfg.use_edl)
        rgb_checkpoint_path = get_path(config['testing'].get('rgb_checkpoint_path',
                                            './models/thumos14/checkpoint-15.ckpt'))
        flow_checkpoint_path = get_path(config['testing'].get('flow_checkpoint_path',
                                             './models/thumos14_flow/checkpoint-16.ckpt'))
        rgb_net.load_state_dict(torch.load(rgb_checkpoint_path))
        flow_net.load_state_dict(torch.load(flow_checkpoint_path))
        rgb_net.eval().cuda()
        flow_net.eval().cuda()
        net = rgb_net
    else:
        net = BDNet(in_channels=config['model']['in_channels'],
                    training=False, use_edl=cfg.use_edl)
        checkpoint_path = get_path(config['testing']['checkpoint_path'])
        net.load_state_dict(torch.load(checkpoint_path))
        net.eval().cuda()
    return net, flow_net

def to_array(data_dict):
    if data_dict is None:
        return []
    result_dict = {}
    for k, v in data_dict.items():
        if v is not None:
            result_dict[k] = v.cpu().numpy()
        else:
            result_dict[k] = v
    return result_dict

def to_tensor(data_dict):
    if len(data_dict) == 0:
        return None
    result_dict = {}
    for k, v in data_dict.items():
        if v is not None:
            result_dict[k] = torch.from_numpy(v)
        else:
            result_dict[k] = v
    return result_dict

def all_to_tensors(outputs):
    for vid, out in enumerate(outputs):
        for sid, (out_rgb, out_flow) in enumerate(zip(out['rgb_out'], out['flow_out'])):
            outputs[vid]['rgb_out'][sid] = to_tensor(out_rgb)
            outputs[vid]['flow_out'][sid] = to_tensor(out_flow)
    return outputs



def get_raw_output(cfg, subset='train'):
    # get data
    node = 'training' if subset == 'train' else 'testing'
    video_infos = get_video_info(config['dataset'][node]['video_info_path'])
    npy_data_path = cfg.rgb_data_path[node] if cfg.fusion else config['dataset'][node]['video_data_path']

    # prepare model
    net, flow_net = build_model(fusion=cfg.fusion)

    centor_crop = videotransforms.CenterCrop(config['dataset']['testing']['crop_size'])
    results = []
    # for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0, desc='Thresholding from Train Set'):
    for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0, desc=f'Inference on {subset} set'):
        # get the clip offsets
        offsetlist = get_offsets(video_infos, video_name, cfg.clip_length, cfg.stride)
        sample_fps = video_infos[video_name]['sample_fps']

        # load data
        data = prepare_data(npy_data_path, video_name, centor_crop)
        flow_data = prepare_data(cfg.flow_data_path, video_name, centor_crop) if cfg.fusion else None

        out_rgb, out_flow = [], []
        for offset in offsetlist:
            # prepare clip of a video
            clip = prepare_clip(data, offset, cfg.clip_length)
            flow_clip = prepare_clip(flow_data, offset, cfg.clip_length) if cfg.fusion else None
            # run inference
            with torch.no_grad():
                output_dict = net(clip)
                flow_output_dict = flow_net(flow_clip) if cfg.fusion else None
            # tensor to numpy array
            out_rgb.append(to_array(output_dict))
            out_flow.append(to_array(flow_output_dict))
        # gather necessary results
        output_video = {'name': video_name, 'fps': sample_fps, 'offset': offsetlist, 'rgb_out': out_rgb, 'flow_out': out_flow}
        results.append(output_video)
    
    return results


def get_all_annos(subset='train', clip_length=256, stride=128):
    node = 'training' if subset == 'train' else 'testing'
    video_infos = get_video_info(config['dataset'][node]['video_info_path'])
    video_annos = get_video_anno(video_infos, config['dataset'][node]['video_anno_path'], config['dataset']['class_info_path'])
    all_annos = []
    # loop for each video
    for video_name in tqdm.tqdm(list(video_annos.keys()), ncols=0, desc=f'Get annotations of {subset} set'):
        annos = video_annos[video_name]
        offsetlist = get_offsets(video_infos, video_name, clip_length, stride)
        for offset in offsetlist:
            # get the annos of the current offset
            left, right = offset + 1, offset + clip_length
            cur_annos = []
            save_offset = False
            for anno in annos:
                max_l = max(left, anno[0])
                min_r = min(right, anno[1])
                ioa = (min_r - max_l) * 1.0 / (anno[1] - anno[0])
                if ioa >= 1.0:
                    save_offset = True
                if ioa >= 0.5:
                    cur_annos.append([max(anno[0] - offset, 1),
                                      min(anno[1] - offset, clip_length),
                                      anno[2]])
            if save_offset:
                start = np.zeros([clip_length])
                end = np.zeros([clip_length])
                for anno in cur_annos:
                    s, e, id = anno
                    d = max((e - s) / 10.0, 2.0)
                    start_s = np.clip(int(round(s - d / 2.0)), 0, clip_length - 1)
                    start_e = np.clip(int(round(s + d / 2.0)), 0, clip_length - 1) + 1
                    start[start_s: start_e] = 1
                    end_s = np.clip(int(round(e - d / 2.0)), 0, clip_length - 1)
                    end_e = np.clip(int(round(e + d / 2.0)), 0, clip_length - 1) + 1
                    end[end_s: end_e] = 1
                all_annos.append({
                    'video_name': video_name,
                    'offset': offset,
                    'annos': cur_annos,
                    'start': start,
                    'end': end
                })
    return all_annos


def compute_iou(pred, target):
    """
    jaccard: A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    """
    pred_left = pred[:, 0]
    pred_right = pred[:, 1]
    target_left = target[:, 0]
    target_right = target[:, 1]

    pred_area = pred_left + pred_right
    target_area = target_left + target_right

    eps = torch.finfo(torch.float32).eps

    inter = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    area_union = target_area + pred_area - inter
    ious = inter / area_union.clamp(min=eps)
    return ious


def get_matched_targets(targets, loc_data, priors, clip_length):
    num_batch = loc_data.size(0)
    num_priors = priors.size(0)
    # match priors and ground truth segments
    loc_t = torch.Tensor(num_batch, num_priors, 2).to(loc_data.device)
    conf_t = torch.LongTensor(num_batch, num_priors).to(loc_data.device)
    prop_loc_t = torch.Tensor(num_batch, num_priors, 2).to(loc_data.device)
    prop_conf_t = torch.LongTensor(num_batch, num_priors).to(loc_data.device)
    for idx in range(num_batch):
        truths = targets[idx][:, :-1]
        labels = targets[idx][:, -1]
        pre_loc = loc_data[idx]
        K = priors.size(0)
        N = truths.size(0)
        center = priors[:, 0].unsqueeze(1).expand(K, N)
        left = (center - truths[:, 0].unsqueeze(0).expand(K, N)) * clip_length
        right = (truths[:, 1].unsqueeze(0).expand(K, N) - center) * clip_length
        area = left + right
        maxn = clip_length * 2
        area[left < 0] = maxn
        area[right < 0] = maxn
        best_truth_area, best_truth_idx = area.min(1)

        loc_t[idx][:, 0] = (priors[:, 0] - truths[best_truth_idx, 0]) * clip_length
        loc_t[idx][:, 1] = (truths[best_truth_idx, 1] - priors[:, 0]) * clip_length
        conf = labels[best_truth_idx]
        conf[best_truth_area >= maxn] = 0
        conf_t[idx] = conf

        iou = compute_iou(pre_loc, loc_t[idx])  # [num_priors]
        prop_conf = conf.clone()
        prop_conf[iou < cfg.overlap_thresh] = 0
        prop_conf_t[idx] = prop_conf
        prop_w = pre_loc[:, 0] + pre_loc[:, 1]
        prop_loc_t[idx][:, 0] = (loc_t[idx][:, 0] - pre_loc[:, 0]) / (0.5 * prop_w)
        prop_loc_t[idx][:, 1] = (loc_t[idx][:, 1] - pre_loc[:, 1]) / (0.5 * prop_w)
    
    return loc_t, conf_t, prop_loc_t, prop_conf_t


def compute_threshold(result_dict):
    all_scores = []
    for vid, proposal_list in result_dict.items():
        for prop in proposal_list:
            if cfg.scoring == 'uncertainty':
                ood_score = 1 - prop['uncertainty']
            elif cfg.scoring == 'confidence':
                ood_score = prop['score']
            elif cfg.scoring == 'uncertainty_actionness':
                ood_score = 1 - prop['uncertainty'] * prop['actionness']
            all_scores.append(ood_score)
    score_sorted = np.sort(all_scores)  # sort the confidence score in an increasing order
    N = len(all_scores)
    topK = N - int(N * 0.95)
    threshold = score_sorted[topK-1]
    return threshold


def post_process(inference_result, phase='train'):
    # send the loaded results into GPU
    inference_result = all_to_tensors(inference_result)

    out_layer = DirichletLayer(evidence=cfg.evidence, dim=-1) if cfg.use_edl else nn.Softmax(dim=-1)
    class_range = range(1, cfg.num_classes) if not cfg.os_head else range(0, cfg.num_classes)
    _, idx_to_class = get_class_index_map(cfg.cls_idx_known)

    result_dict = {}
    for out in tqdm.tqdm(inference_result, total=len(inference_result), desc=f'{phase} phase post-processing'):
        video_name = out['name']
        sample_fps = out['fps']
        output = [[] for cl in range(cfg.num_classes)]
        # post-processing
        for (out_rgb, out_flow, offset) in zip(out['rgb_out'], out['flow_out'], out['offset']):
            # out_rgb, out_flow = to_tensor(out_rgb), to_tensor(out_flow)
            loc, conf, prop_loc, prop_conf, center, priors, unct, prop_unct, act, prop_act = parse_output(out_rgb, out_flow, fusion=cfg.fusion, use_edl=cfg.use_edl, os_head=cfg.os_head)
            decoded_segments, conf_scores, uncertainty, actionness = decode_predictions(loc, prop_loc, priors, conf, prop_conf, unct, prop_unct, act, prop_act, \
                                                                center, offset, sample_fps, cfg.clip_length, cfg.num_classes, score_func=out_layer, use_edl=cfg.use_edl, os_head=cfg.os_head)
            # filtering out clip-level predictions with low confidence
            for cl in class_range:  # from 1 to K+1 by default, or 0 to K for os_head
                segments = filtering(decoded_segments, conf_scores[cl], uncertainty, actionness, cfg.conf_thresh, use_edl=cfg.use_edl, os_head=cfg.os_head)  # (N,5)
                if segments is None:
                    continue
                output[cl].append(segments)
            
        # get final detection results for each video
        result_dict[video_name] = get_video_detections(output, idx_to_class, cfg.num_classes, cfg.top_k, cfg.nms_sigma, use_edl=cfg.use_edl, os_head=cfg.os_head, cls_rng=class_range)

    if phase == 'train':
        # get the score threshold
        threshold = compute_threshold(result_dict)
        return threshold
    else:
        # temporarily save the results for later evaluation
        output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {}}
        temp_dir = os.path.join(cfg.output_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        pred_file = os.path.join(temp_dir, f'thumos14_open_rgb-test.json')
        with open(pred_file, "w") as f:
            json.dump(output_dict, f)
        return result_dict, pred_file
    

def split_uncertainties_stages(output_test, annos_known_test):
    all_known_uncertainty, all_unknown_uncertainty = {'coarse': [], 'refined': []}, {'coarse': [], 'refined': []}
    for output_video in output_test:
        out_rgb = output_video['rgb_out']
        offsetlist = output_video['offset']
        video_name = output_video['name']
        annos_cur_video = [anno for anno in annos_known_test if video_name in anno.values()]
        if len(annos_cur_video) == 0:  # clips in this video are all unknown/bg
            all_unknown_uncertainty['coarse'].extend([clip_out['unct'][0] for clip_out in out_rgb])
            all_unknown_uncertainty['refined'].extend([clip_out['prop_unct'][0] for clip_out in out_rgb])
            continue
        # the rest videos contain at least one known action
        for clip_out, offset in zip(out_rgb, offsetlist):  # iterate on clips
            # get the matched target label
            annos_cur_clip = [anno['annos'] for anno in annos_cur_video if offset == anno['offset']]
            if len(annos_cur_clip) == 0:
                # current clip is unknown/bg
                all_unknown_uncertainty['coarse'].append(clip_out['unct'][0])
                all_unknown_uncertainty['refined'].append(clip_out['prop_unct'][0])
                continue

            # the rest are used in training (known or unknown/bg)
            annos = annos_transform(annos_cur_clip[0], cfg.clip_length)
            targets = [torch.from_numpy(np.stack(annos, 0))]
            loc_t, conf_t, prop_loc_t, prop_conf_t = get_matched_targets(
                targets, torch.from_numpy(clip_out['loc']), 
                torch.from_numpy(clip_out['priors']), cfg.clip_length)
            # coarse stage
            inds_pos = conf_t.view(-1) > 0
            inds_neg = conf_t.view(-1) <= 0
            all_known_uncertainty['coarse'].append(clip_out['unct'][0][inds_pos])
            all_unknown_uncertainty['coarse'].append(clip_out['unct'][0][inds_neg])
            # refined stage
            inds_pos = prop_conf_t.view(-1) > 0
            inds_neg = prop_conf_t.view(-1) <= 0
            all_known_uncertainty['refined'].append(clip_out['prop_unct'][0][inds_pos])
            all_unknown_uncertainty['refined'].append(clip_out['prop_unct'][0][inds_neg])
    return all_known_uncertainty, all_unknown_uncertainty


def plot_unct_dist(result_file, all_uncertainty, colors, labels):

    fig = plt.figure(figsize=(5,4))  # (w, h)
    fontsize = 18
    plt.rcParams["font.family"] = "Arial"
    plt.hist(all_uncertainty, 100, density=False, color=colors, label=labels)
    plt.legend(fontsize=fontsize-3)
    plt.xlabel('vacuity uncertainty', fontsize=fontsize)
    plt.ylabel('number of predictions', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.xlim(0, 1.01)
    # plt.ylim(0, 40000)
    plt.tight_layout()
    plt.savefig(result_file)


def get_activity_index(class_info_path):
    txt = np.loadtxt(class_info_path, dtype=str)
    class_to_idx = {}
    if cfg.open_set:
        class_to_idx['__unknown__'] = 0  # 0 is reserved for unknown in open set
    for idx, l in enumerate(txt):
        class_to_idx[l[1]] = idx + 1  # starting from 1 to K (K=15 for thumos14)
    return class_to_idx
    

def load_gt_data(ground_truth_filename, activity_index):
    with open(ground_truth_filename, 'r') as fobj:
        data = json.load(fobj)
    # Read ground truth data.
    video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
    for videoid, v in data['database'].items():
        for ann in v['annotations']:
            video_lst.append(videoid)
            t_start_lst.append(float(ann['segment'][0]))
            t_end_lst.append(float(ann['segment'][1]))
            if cfg.open_set:
                if ann['label'] in activity_index:
                    label_lst.append(activity_index[ann['label']])
                else:
                    label_lst.append(0)  # the unknown
            else:  # closed set
                assert ann['label'] in activity_index, 'Ground truth json contains invalid class: %s'%(ann['label'])
                label_lst.append(activity_index[ann['label']])
    
    ground_truth = pd.DataFrame({'video-id': video_lst, 
                    't-start': t_start_lst,
                    't-end': t_end_lst,
                    'label': label_lst})
    return ground_truth, video_lst


def gather_valid_preds(result_dict, video_lst, activity_index):
    predictions = {'video-id': [], 't-start': [], 't-end': [], 'label': [], 'uncertainty': []}
    for videoid, proposal_list in result_dict.items():
        if videoid not in video_lst:
            continue
        for result in proposal_list:
            if result['label'] not in activity_index:
                continue
            # known/unknown classification
            if cfg.scoring == 'uncertainty':
                res_score = 1 - result['uncertainty']
            elif cfg.scoring == 'confidence':
                res_score = result['score']
            elif cfg.scoring == 'uncertainty_actionness':
                res_score = 1 - result['uncertainty'] * result['actionness']
            if cfg.open_set and res_score < ood_thresh:
                label = activity_index['__unknown__']  # reject the unknown
            else:
                label = activity_index[result['label']]
            
            predictions['video-id'].append(videoid)
            predictions['t-start'].append(float(result['segment'][0]))
            predictions['t-end'].append(float(result['segment'][1]))
            predictions['label'].append(label)
            predictions['uncertainty'].append(result['uncertainty'])
    predictions = pd.DataFrame(predictions)
    return predictions


def split_uncertainties(prediction_all, ground_truth_all, video_lst, tiou_thr=0.5):
    ground_truth_by_vid = ground_truth_all.groupby('video-id')
    prediction_by_vid = prediction_all.groupby('video-id')
    def _get_predictions_with_vid(prediction_by_vid, video_name):
        try:
            return prediction_by_vid.get_group(video_name).reset_index(drop=True)
        except:
            return pd.DataFrame()
    known_uncertainty, unknown_uncertainty, background_uncertainty = [], [], []
    for video_name in tqdm.tqdm(video_lst, total=len(video_lst)):
        ground_truth = ground_truth_by_vid.get_group(video_name).reset_index()
        prediction = _get_predictions_with_vid(prediction_by_vid, video_name)
        if prediction.empty:
            continue
        lock_gt = np.ones((len(ground_truth))) * -1
        for idx, this_pred in prediction.iterrows():
            uncertainty = this_pred['uncertainty']
            tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                                   ground_truth[['t-start', 't-end']].values)
            tiou_sorted_idx = tiou_arr.argsort()[::-1]  # tIoU in a decreasing order
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:  # background segment
                    background_uncertainty.append(uncertainty)
                    break
                if lock_gt[jdx] >= 0:
                    continue  # this gt was matched before, continue to select the second largest tIoU match
                label_gt = int(ground_truth.loc[jdx]['label'])
                if label_gt == 0: # unknown foreground
                    unknown_uncertainty.append(uncertainty)
                else:  # known foreground
                    known_uncertainty.append(uncertainty)
                lock_gt[jdx] = idx
                break
    return known_uncertainty, unknown_uncertainty, background_uncertainty



if __name__ == '__main__':

    cfg = get_basic_config(config)
    
    output_file = os.path.join(cfg.output_path, 'raw_outputs.npz')
    if not os.path.exists(output_file):
        output_train = get_raw_output(cfg, subset='train')
        output_test = get_raw_output(cfg, subset='test')
        np.savez(output_file[:-4], train=output_train, test=output_test)
    else:
        results = np.load(output_file, allow_pickle=True)
        output_train, output_test = results['train'], results['test']
        print(f'Raw output result file already exist at {output_file}!')

    # get the annotations of all video clips
    # annos_train = get_all_annos(subset='train', clip_length=cfg.clip_length, stride=cfg.stride)
    annos_known_test = get_all_annos(subset='test', clip_length=cfg.clip_length, stride=cfg.stride)

    all_known_uncertainty, all_unknown_uncertainty = split_uncertainties_stages(output_test, annos_known_test)

    # draw distributions before post-processing
    fig_dir = os.path.join(cfg.output_path, 'dist')
    os.makedirs(fig_dir, exist_ok=True)
    # draw ood uncertainty distribution in coarse stage
    ind_uncertainty = np.concatenate(all_known_uncertainty['coarse'])
    ood_uncertainty = np.concatenate(all_unknown_uncertainty['coarse'])
    plot_unct_dist(os.path.join(fig_dir, 'unct_dist_coarse.png'), [ind_uncertainty, ood_uncertainty],
        colors=['green', 'red'], labels=['Known', 'Unknown'])

    # draw ood uncertainty distribution in refined stage
    ind_uncertainty = np.concatenate(all_known_uncertainty['refined'])
    ood_uncertainty = np.concatenate(all_unknown_uncertainty['refined'])
    plot_unct_dist(os.path.join(fig_dir, 'unct_dist_refined.png'), [ind_uncertainty, ood_uncertainty],
        colors=['green', 'red'], labels=['Known', 'Unknown'])
    
    # get the threshold from trainset inference results
    ood_thresh = post_process(output_train, phase='train')
    # get the post process results for evaluation
    result_dict, pred_file = post_process(output_test, phase='test')

    activity_index = get_activity_index(cfg.cls_idx_known)

    gt_file = cfg.gt_all_json if cfg.open_set else cfg.gt_known_json.format(id=cfg.split)
    ground_truth_all, video_lst = load_gt_data(gt_file, activity_index)

    prediction_all = gather_valid_preds(result_dict, video_lst, activity_index)

    known_uncertainty, unknown_uncertainty, background_uncertainty = split_uncertainties(prediction_all, ground_truth_all, video_lst, tiou_thr=0.5)
    # plot 
    plot_unct_dist(os.path.join(fig_dir, 'unct_dist_final.png'), [known_uncertainty, unknown_uncertainty, background_uncertainty],
        colors=['green', 'red', 'blue'], labels=['Known', 'Unknown', 'Background'])
    plot_unct_dist(os.path.join(fig_dir, 'unct_dist_nobg.png'), [known_uncertainty, unknown_uncertainty],
        colors=['green', 'red'], labels=['Known', 'Unknown'])