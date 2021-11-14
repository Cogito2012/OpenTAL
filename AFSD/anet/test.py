import torch
import torch.nn as nn
import os
import numpy as np
import tqdm
import json
from AFSD.common import videotransforms
from AFSD.common.anet_dataset import get_video_info, load_json
from AFSD.anet.BDNet import BDNet, DirichletLayer
from AFSD.common.segment_utils import softnms_v2
from AFSD.common.config import config

import multiprocessing as mp
import threading



def get_basic_config(config, dataset='testing'):
    class cfg: pass
    cfg.num_classes = config['dataset']['num_classes']
    cfg.conf_thresh = config['testing']['conf_thresh']
    cfg.top_k = config['testing']['top_k']
    cfg.nms_thresh = config['testing']['nms_thresh']
    cfg.nms_sigma = config['testing']['nms_sigma']
    cfg.clip_length = config['dataset']['testing']['clip_length']
    cfg.stride = config['dataset']['testing']['clip_stride']
    cfg.crop_size = config['dataset']['testing']['crop_size']
    cfg.input_channels = config['model']['in_channels']
    cfg.checkpoint_path = config['testing']['checkpoint_path']
    cfg.use_edl = config['model']['use_edl'] if 'use_edl' in config['model'] else False
    if cfg.use_edl:
        cfg.evidence = config['model']['evidence']
    cfg.scoring = config['testing']['ood_scoring']
    cfg.os_head = config['model']['os_head'] if 'os_head' in config['model'] else False
    if cfg.os_head:
        cfg.num_classes = cfg.num_classes - 1
    
    cfg.json_name = config['testing']['output_json']
    cfg.output_path = config['testing']['output_path']
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path)
    cfg.fusion = config['testing']['fusion']
    cfg.ngpu = config['ngpu']
    cfg.thread_num = config['ngpu']

    subset = 'validation' if dataset == 'testing' else 'training'
    cfg.idx_to_class = get_class_names(config['dataset']['class_info_path'])
    cfg.video_infos = get_video_info(config['dataset'][dataset]['video_info_path'], subset=subset)
    cfg.mp4_data_path = config['dataset'][dataset]['video_mp4_path']  # train_val_npy_112/

    return cfg


def get_class_names(class_info_path):
    idx_to_class = {}
    with open(class_info_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            idx_to_class[idx + 1] = line.strip()    # starting from 1 to K (K=150 for activitynet)
    return idx_to_class


def get_path(input_path):
    if os.path.lexists(input_path):
        fullpath = os.path.realpath(input_path) if os.path.islink(input_path) else input_path
        real_name = fullpath.split('/')[-1]
        real_full_path = os.path.join(os.path.dirname(input_path), real_name)
    else:
        raise FileNotFoundError
    return real_full_path



def prepare_data(video_name, center_crop, cfg):
    sample_fps = cfg.video_infos[video_name]['fps']
    duration = cfg.video_infos[video_name]['duration']
    offsetlist = [0]
    # get video data
    data = np.load(os.path.join(cfg.mp4_data_path, video_name + '.npy'))
    frames = data
    frames = np.transpose(frames, [3, 0, 1, 2])
    data = center_crop(frames)
    data = torch.from_numpy(data.copy()).cuda()
    return data, offsetlist, sample_fps, duration


def prepare_clip(data, offset, clip_length, crop_size):
    clip = data[:, offset: offset + clip_length]
    clip = clip.float()
    if clip.size(1) < clip_length:
        tmp = torch.ones(
            [clip.size(0), clip_length - clip.size(1), crop_size, crop_size]).float().cuda() * 127.5
        clip = torch.cat([clip, tmp], dim=1)
    clip = clip.unsqueeze(0)
    clip = (clip / 255.0) * 2.0 - 1.0
    return clip


def decode_prediction(output_dict, cfg, score_func=nn.Softmax(dim=-1)):
    # batchsize should be 1!
    loc, conf, priors = output_dict['loc'][0], output_dict['conf'][0], output_dict['priors']
    prop_loc, prop_conf = output_dict['prop_loc'][0], output_dict['prop_conf'][0]
    center = output_dict['center'][0]
    # conditional outputs
    act = output_dict['act'][0].squeeze(-1) if cfg.os_head else None
    prop_act = output_dict['prop_act'][0].squeeze(-1) if cfg.os_head else None
    unct = output_dict['unct'][0] if cfg.use_edl else None
    prop_unct = output_dict['prop_unct'][0] if cfg.use_edl else None

    # decode the locations of segments
    pre_loc_w = loc[:, :1] + loc[:, 1:]
    loc = 0.5 * pre_loc_w * prop_loc + loc
    decoded_segments = torch.cat(
        [priors[:, :1] * cfg.clip_length - loc[:, :1],
            priors[:, :1] * cfg.clip_length + loc[:, 1:]], dim=-1)
    decoded_segments.clamp_(min=0, max=cfg.clip_length)

    # compute uncertainty and actionness
    uncertainty = (unct + prop_unct) / 2.0 if cfg.use_edl else None
    actionness = (act.sigmoid() + prop_act.sigmoid()) / 2.0 if cfg.os_head else None

    # compute classification confidence
    conf = score_func(conf)
    prop_conf = score_func(prop_conf)
    center = center.sigmoid()

    conf = (conf + prop_conf) / 2.0
    conf = conf * center * actionness.unsqueeze(-1) if cfg.os_head else conf * center
    conf = conf.view(-1, cfg.num_classes).transpose(1, 0)
    conf_scores = conf.clone()

    return decoded_segments, conf_scores, uncertainty, actionness


def filtering(decoded_segments, conf_score_cls, uncertainty, actionness, offset, sample_fps, cfg, conf_thresh=0.001):
    c_mask = conf_score_cls > conf_thresh
    if cfg.os_head:
        c_mask = c_mask & (actionness > 0.5)
    scores = conf_score_cls[c_mask]
    if scores.size(0) == 0:
        return None
    l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
    segments = decoded_segments[l_mask].view(-1, 2)
    segments = (segments + offset) / sample_fps
    segments = torch.cat([segments, scores.unsqueeze(1)], -1)  # (N, 3)
    if cfg.use_edl:
        # masking uncertainties
        uncertain_scores = uncertainty[c_mask]
        segments = torch.cat([segments, uncertain_scores.unsqueeze(1)], -1)  # (N, 4)
    if cfg.os_head:
        # masking actionness
        act_scores = actionness[c_mask]
        segments = torch.cat([segments, act_scores.unsqueeze(1)], -1)  # (N, 4) or (N, 5)
    return segments


def get_video_prediction(output, duration, cfg, cls_rng=None):
    res_dim = 3
    res_dim = res_dim + 1 if cfg.use_edl else res_dim  # 3 or 4
    res_dim = res_dim + 1 if cfg.os_head else res_dim  # 3 or 4 or 5
    res = torch.zeros(cfg.num_classes, cfg.top_k, res_dim)
    # NMS for each class
    for cl in cls_rng:
        if len(output[cl]) == 0:
            continue
        tmp = torch.cat(output[cl], 0)
        tmp, count = softnms_v2(tmp, sigma=cfg.nms_sigma, top_k=cfg.top_k, score_threshold=0.001, use_edl=cfg.use_edl, os_head=cfg.os_head)
        res[cl, :count] = tmp

    flt = res.contiguous().view(-1, res_dim)
    flt = flt.view(cfg.num_classes, -1, res_dim)
    proposal_list = []
    for cl in cls_rng:
        cl_idx = cl + 1 if cfg.os_head else cl
        class_name = cfg.idx_to_class[cl_idx]  # assume the current video contains only one class
        tmp = flt[cl].contiguous()
        tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, res_dim)
        if tmp.size(0) == 0:
            continue
        tmp = tmp.detach().cpu().numpy()
        for i in range(tmp.shape[0]):
            tmp_proposal = {}
            start_time = max(0, float(tmp[i, 0]))
            end_time = min(duration, float(tmp[i, 1]))
            if end_time <= start_time:
                continue
            tmp_proposal['label'] = class_name
            tmp_proposal['score'] = float(tmp[i, 2])
            tmp_proposal['segment'] = [start_time, end_time]
            tmp_proposal['uncertainty'] = float(tmp[i, 3]) if cfg.use_edl else 0.0
            tmp_proposal['actionness'] = float(tmp[i, 4]) if cfg.os_head else 0.0
            proposal_list.append(tmp_proposal)
    return proposal_list


def inference_thread(lock, pid, video_list, cfg, result_dict):

    torch.cuda.set_device(pid)
    net = BDNet(in_channels=cfg.input_channels,
                training=False, use_edl=cfg.use_edl)
    net.load_state_dict(torch.load(get_path(cfg.checkpoint_path)))
    net.eval().cuda()

    out_layer = DirichletLayer(evidence=cfg.evidence, dim=-1) if cfg.use_edl else nn.Softmax(dim=-1)
    center_crop = videotransforms.CenterCrop(cfg.crop_size)
    class_range = range(1, cfg.num_classes) if not cfg.os_head else range(0, cfg.num_classes)

    text = 'processor %d' % pid
    with lock:
        progress = tqdm.tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    for video_name in video_list:
        # get video information
        data, offsetlist, sample_fps, duration = prepare_data(video_name, center_crop, cfg)
        
        output = [[] for cl in range(cfg.num_classes)]
        for offset in offsetlist:
            # prepare clip of a video
            clip = prepare_clip(data, offset, cfg.clip_length, cfg.crop_size)

            # run inference
            with torch.no_grad():
                output_dict = net(clip)

            # decode results
            decoded_segments, conf_scores, uncertainty, actionness = decode_prediction(output_dict, cfg, out_layer)
            # filtering
            for cl in class_range:
                segments = filtering(decoded_segments, conf_scores[cl], uncertainty, actionness, offset, sample_fps, cfg)
                if segments is None:
                    continue
                output[cl].append(segments)
        # finish offset loop

        # get final detection results for each video
        result_dict[video_name[2:]] = get_video_prediction(output, duration, cfg, cls_rng=class_range)
        with lock:
            progress.update(1)
    # finish video loop

    with lock:
        progress.close()


def testing_multithread(cfg, video_list, output_file, thread_num=1):
    processes = []
    lock = threading.Lock()

    video_num = len(video_list)
    per_thread_video_num = video_num // thread_num
    result_dict = mp.Manager().dict()

    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        # inference_thread(lock, i, sub_video_list, test_cls_data, cfg)
        p = mp.Process(target=inference_thread, args=(lock, i, sub_video_list, cfg, result_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # save results
    assert len(result_dict) == video_num, "Incomplete testing results!"
    output_dict = {"version": "ActivityNet-v1.3", "results": dict(result_dict), "external_data": {}}
    with open(output_file, "w") as out:
        json.dump(output_dict, out)


def testing(cfg, video_list, output_file):

    net = BDNet(in_channels=cfg.input_channels,
                training=False, use_edl=cfg.use_edl)
    net.load_state_dict(torch.load(get_path(cfg.checkpoint_path)))
    net.eval().cuda()

    out_layer = DirichletLayer(evidence=cfg.evidence, dim=-1) if cfg.use_edl else nn.Softmax(dim=-1)
    center_crop = videotransforms.CenterCrop(cfg.crop_size)
    class_range = range(1, cfg.num_classes) if not cfg.os_head else range(0, cfg.num_classes)

    result_dict = {}
    progress = tqdm.tqdm(
        total=len(video_list),
        position=0,
        desc='ANet Inference',
        ncols=0
    )
    for video_name in video_list:
        # get video information
        data, offsetlist, sample_fps, duration = prepare_data(video_name, center_crop, cfg)
        
        output = [[] for cl in range(cfg.num_classes)]
        for offset in offsetlist:
            # prepare clip of a video
            clip = prepare_clip(data, offset, cfg.clip_length, cfg.crop_size)
            # run inference
            with torch.no_grad():
                output_dict = net(clip)

            # decode results
            decoded_segments, conf_scores, uncertainty, actionness = decode_prediction(output_dict, cfg, out_layer)
            # filtering
            for cl in class_range:
                segments = filtering(decoded_segments, conf_scores[cl], uncertainty, actionness, offset, sample_fps, cfg)
                if segments is None:
                    continue
                output[cl].append(segments)
        # get final detection results for each video
        result_dict[video_name[2:]] = get_video_prediction(output, duration, cfg, cls_rng=class_range)
        progress.update(1)

    # save results
    assert len(result_dict) == len(video_list), "Incomplete testing results!"
    output_dict = {"version": "ActivityNet-v1.3", "results": dict(result_dict), "external_data": {}}
    with open(output_file, "w") as out:
        json.dump(output_dict, out)


def main():
    cfg = get_basic_config(config, dataset='testing')

    video_list = list(cfg.video_infos.keys())
    videos_in_disk = [filename[:-4] for filename in os.listdir(cfg.mp4_data_path)]
    video_list = list(set(video_list) & set(videos_in_disk))
    
    output_file = os.path.join(cfg.output_path, cfg.json_name)
    # testing_multithread(cfg, video_list, output_file, thread_num=cfg.thread_num)

    rerun = True
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            output_dict = json.load(f)
        if len(output_dict['results']) == len(video_list):
            rerun = False
    if rerun:
        testing(cfg, video_list, output_file)
    else:
        print(f'Result file exist and it is complete! \n{output_file}')


if __name__ == '__main__':
    # keep all things private in this file
    main()