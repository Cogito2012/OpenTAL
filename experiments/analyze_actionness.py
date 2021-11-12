from re import search
from matplotlib import colors
import torch
import torch.nn as nn
import os, sys
import numpy as np
import tqdm
import json
from AFSD.common import videotransforms
from AFSD.common.thumos_dataset import get_video_info, get_class_index_map, annos_transform
from AFSD.thumos14.BDNet import BDNet, DirichletLayer
from AFSD.thumos14.test import get_offsets, prepare_data, prepare_clip, parse_output, decode_predictions, filtering, get_video_detections, get_path
from AFSD.common.config import config
from AFSD.evaluation.eval_detection import ANETdetection
import matplotlib.pyplot as plt
import pandas as pd
from AFSD.evaluation.utils_eval import segment_iou
import seaborn as sns


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
    cfg.cls_idx_all = f'datasets/thumos14/annotations_open/split_{cfg.split}/Class_Index_All.txt'

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


def get_video_anno(video_infos, video_anno_path, class_info_path):
    df_anno = pd.DataFrame(pd.read_csv(video_anno_path)).values[:]
    originidx_to_idx, idx_to_class = get_class_index_map(class_info_path)
    video_annos = {}
    for anno in df_anno:
        video_name = anno[0]
        originidx = anno[2]
        start_frame = anno[-2]
        end_frame = anno[-1]
        count = video_infos[video_name]['count']
        sample_count = video_infos[video_name]['sample_count']
        ratio = sample_count * 1.0 / count
        start_gt = start_frame * ratio
        end_gt = end_frame * ratio
        class_idx = originidx_to_idx[originidx]
        if video_annos.get(video_name) is None:
            video_annos[video_name] = [[start_gt, end_gt, class_idx]]
        else:
            video_annos[video_name].append([start_gt, end_gt, class_idx])
    return video_annos


def get_all_annos(subset='train', clip_length=256, stride=128):
    node = 'training' if subset == 'train' else 'testing'
    video_infos = get_video_info(config['dataset'][node]['video_info_path'])
    video_annos = get_video_anno(video_infos, config['dataset'][node]['video_anno_open_path'], cfg.cls_idx_all)
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


def get_result(input_dict, stage='coarse', target='uncertainty'):
    if cfg.use_edl:
        unct = input_dict['unct'][0] if stage == 'coarse' else input_dict['prop_unct'][0]  # (N,)
    if cfg.os_head:
        act = input_dict['act'][0] if stage == 'coarse' else input_dict['prop_act'][0]  # (N, 1)
        act = torch.from_numpy(act).sigmoid().numpy()
    
    # get the output target
    if target == 'uncertainty' and cfg.use_edl:
        return unct
    if target == 'actionness' and cfg.use_edl and cfg.open_set:
        return np.squeeze(act, axis=-1)
    elif target == 'confidence':
        out_layer = DirichletLayer(evidence=cfg.evidence, dim=-1) if cfg.use_edl else nn.Softmax(dim=-1)
        # get the uncertainty, actionness, and conf_scores
        logits = input_dict['conf'][0] if stage == 'coarse' else input_dict['prop_conf'][0]  # N x K
        conf = out_layer(torch.from_numpy(logits))
        center = torch.from_numpy(input_dict['center'][0])
        conf = conf * center.sigmoid()  # N x K
        conf = conf.numpy()
        if cfg.os_head:
            conf = conf * act
        conf = np.max(conf, axis=-1)  # (N,)
        return conf
    elif target == 'uncertainty_actionness' and cfg.use_edl:
        return unct * np.squeeze(act, axis=-1)  # (N,)
    elif target == 'half_au' and cfg.use_edl:
        return 0.5 * (np.squeeze(act, axis=-1) + 1.0) * unct
    else:
        raise NotImplementedError


def split_results_by_stages(output_test, annos_known_test, target='uncertainty'):

    all_known, all_unknown, all_bg = {'coarse': [], 'refined': []}, {'coarse': [], 'refined': []}, {'coarse': [], 'refined': []}
    for output_video in output_test:
        out_rgb = output_video['rgb_out']
        offsetlist = output_video['offset']
        video_name = output_video['name']
        annos_cur_video = [anno for anno in annos_known_test if video_name in anno.values()]
        if len(annos_cur_video) == 0:  # ignore if the video is not annotated
            continue
        # the rest videos contain at least one known action
        for clip_out, offset in zip(out_rgb, offsetlist):  # iterate on clips
            annos_cur_clip = [anno['annos'] for anno in annos_cur_video if offset == anno['offset']]
            if len(annos_cur_clip) == 0:  # current clip is unknown/bg
                continue
            # the rest are used in training (known or unknown/bg)
            annos = annos_transform(annos_cur_clip[0], cfg.clip_length)
            targets = [torch.from_numpy(np.stack(annos, 0))]
            # get the matched target label
            loc_t, conf_t, prop_loc_t, prop_conf_t = get_matched_targets(
                targets, torch.from_numpy(clip_out['loc']), 
                torch.from_numpy(clip_out['priors']), cfg.clip_length)

            # coarse stage
            conf_t, prop_conf_t = conf_t.view(-1), prop_conf_t.view(-1)
            inds_uk = conf_t > cfg.num_classes  # y > K
            inds_k = (conf_t > 0) & (conf_t <= cfg.num_classes)  # 1 <= y <= K
            inds_bg = conf_t == 0  # y == 0
            res_coarse = get_result(clip_out, stage='coarse', target=target)
            all_known['coarse'].append(res_coarse[inds_k])
            all_unknown['coarse'].append(res_coarse[inds_uk])
            all_bg['coarse'].append(res_coarse[inds_bg])

            # refined stage
            inds_uk = prop_conf_t > cfg.num_classes  # y > K
            inds_k = (prop_conf_t > 0) & (prop_conf_t <= cfg.num_classes)  # 1 <= y <= K
            inds_bg = prop_conf_t == 0  # y == 0
            res_refined = get_result(clip_out, stage='refined', target=target)
            all_known['refined'].append(res_refined[inds_k])
            all_unknown['refined'].append(res_refined[inds_uk])
            all_bg['refined'].append(res_refined[inds_bg])
    return all_known, all_unknown, all_bg


def plot_dist(result_file, all_scores, colors, labels, xlabel=None):

    fig = plt.figure(figsize=(5,4))  # (w, h)
    fontsize = 18
    plt.rcParams["font.family"] = "Arial"
    # plt.hist(all_scores, 100, density=normalize_fig, color=colors, label=labels)
    for score, color, label in zip(all_scores, colors, labels):
        sns.kdeplot(score, color=color, shade=True, label=label)
    plt.legend(fontsize=fontsize-3, loc='upper center')
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
    y_labels = 'density' if normalize_fig else 'number of predictions'
    plt.ylabel(y_labels, fontsize=fontsize)
    # plt.yscale('log')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # if normalize_fig:
    #     plt.xlim(0, 1.01)
    # plt.ylim(0, 40000)
    plt.tight_layout()
    plt.savefig(result_file)


if __name__ == '__main__':

    cfg = get_basic_config(config)
    normalize_fig = True
    target = 'uncertainty'  # 'confidence', 'uncertainty', 'actionness', 'uncertainty_actionness'
    # cfg.output_path = 'output/opental_final/split_0'
    
    output_file = os.path.join(cfg.output_path, 'raw_outputs_test.npz')
    print(output_file)
    if not os.path.exists(output_file):
        output_test = get_raw_output(cfg, subset='test')
        np.savez(output_file[:-4], test=output_test)
    else:
        results = np.load(output_file, allow_pickle=True)
        output_test = results['test']
        print(f'Raw output result file already exist at {output_file}!')

    # get the annotations of all video clips
    annos_known_test = get_all_annos(subset='test', clip_length=cfg.clip_length, stride=cfg.stride)

    all_known, all_unknown, all_bg = split_results_by_stages(output_test, annos_known_test, target=target)

    # draw ood distribution in coarse stage
    ind_data = np.concatenate(all_known['coarse'])
    ood_data = np.concatenate(all_unknown['coarse'])
    bg_data = np.concatenate(all_bg['coarse'])
    # draw distributions before post-processing
    out_folder = 'dist_norm' if normalize_fig else 'dist'
    fig_dir = os.path.join(cfg.output_path, out_folder, target)
    os.makedirs(fig_dir, exist_ok=True)

    plot_dist(os.path.join(fig_dir, 'dist_coarse.png'), [ind_data, ood_data, bg_data],
        colors=['green', 'red', 'cyan'], labels=['Known', 'Unknown', 'Background'], xlabel=target)
    if target == 'actionness':
        fg_data = np.concatenate((ind_data, ood_data))
        plot_dist(os.path.join(fig_dir, 'dist_coarse_act.png'), [fg_data, bg_data],
            colors=['red', 'blue'], labels=['Foreground', 'Background'], xlabel=target)
    elif target == 'uncertainty':
        plot_dist(os.path.join(fig_dir, 'dist_coarse_unct.png'), [ind_data, ood_data],
            colors=['red', 'blue'], labels=['Known Actions', 'Unknown Actions'], xlabel=target)

    # draw ood distribution in refined stage
    ind_data = np.concatenate(all_known['refined'])
    ood_data = np.concatenate(all_unknown['refined'])
    bg_data = np.concatenate(all_bg['refined'])
    # draw distributions before post-processing
    out_folder = 'dist_norm' if normalize_fig else 'dist'
    fig_dir = os.path.join(cfg.output_path, out_folder, target)
    os.makedirs(fig_dir, exist_ok=True)

    plot_dist(os.path.join(fig_dir, 'dist_refined.png'), [ind_data, ood_data, bg_data],
        colors=['darkgreen', 'red', 'cyan'], labels=['Known', 'Unknown', 'Background'], xlabel=target)
    if target == 'actionness':
        fg_data = np.concatenate((ind_data, ood_data))
        plot_dist(os.path.join(fig_dir, 'dist_refined_act.png'), [fg_data, bg_data],
            colors=['red', 'blue'], labels=['Foreground', 'Background'], xlabel=target)
        plot_dist(os.path.join(fig_dir, 'dist_refined_act.pdf'), [fg_data, bg_data],
            colors=['red', 'blue'], labels=['Foreground', 'Background'])
    elif target == 'uncertainty':
        plot_dist(os.path.join(fig_dir, 'dist_refined_unct.png'), [ind_data, ood_data],
            colors=['red', 'blue'], labels=['Known Actions', 'Unknown Actions'], xlabel=target)
        plot_dist(os.path.join(fig_dir, 'dist_refined_unct.pdf'), [ind_data, ood_data],
            colors=['red', 'blue'], labels=['Known Actions', 'Unknown Actions'])