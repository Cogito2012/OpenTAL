import torch
import torch.nn as nn
import os
import numpy as np
import tqdm
import json
from AFSD.common import videotransforms
from AFSD.common.thumos_dataset import get_video_info, get_class_index_map
from AFSD.thumos14.BDNet import BDNet
from test import get_offsets, prepare_data, prepare_clip, parse_output, decode_predictions, filtering, get_video_detections, get_path
from AFSD.common.config import config


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
    cfg.scoring = config['testing']['ood_scoring']
    cfg.os_head = config['model']['os_head'] if 'os_head' in config['model'] else False
    if cfg.os_head:
        cfg.num_classes = cfg.num_classes - 1

    cfg.json_name = config['testing']['output_json']
    cfg.output_path = config['testing']['output_path']
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path)
    cfg.fusion = config['testing']['fusion']

    # getting path for fusion
    cfg.rgb_data_path = config['training'].get('rgb_data_path',
                                        './datasets/thumos14/validation_npy/')
    cfg.flow_data_path = config['training'].get('flow_data_path',
                                        './datasets/thumos14/validation_flow_npy/')
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


def thresholding(cfg, output_file):
    # get data
    video_infos = get_video_info(config['dataset']['training']['video_info_path'])
    _, idx_to_class = get_class_index_map(config['dataset']['class_info_path'])
    npy_data_path = cfg.rgb_data_path if cfg.fusion else config['dataset']['training']['video_data_path']

    # prepare model
    net, flow_net = build_model(fusion=cfg.fusion)

    centor_crop = videotransforms.CenterCrop(config['dataset']['testing']['crop_size'])
    result_dict = {}
    for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0, desc='Thresholding from Train Set'):
        # get the clip offsets
        offsetlist = get_offsets(video_infos, video_name, cfg.clip_length, cfg.stride)
        sample_fps = video_infos[video_name]['sample_fps']

        # load data
        data = prepare_data(npy_data_path, video_name, centor_crop)
        flow_data = prepare_data(cfg.flow_data_path, video_name, centor_crop) if cfg.fusion else None

        output = [[] for cl in range(cfg.num_classes)]
        for offset in offsetlist:
            # prepare clip of a video
            clip = prepare_clip(data, offset, cfg.clip_length)
            flow_clip = prepare_clip(flow_data, offset, cfg.clip_length) if cfg.fusion else None
            # run inference
            with torch.no_grad():
                output_dict = net(clip)
                flow_output_dict = flow_net(flow_clip) if cfg.fusion else None

            loc, conf, prop_loc, prop_conf, center, priors, unct, prop_unct, act, prop_act = parse_output(output_dict, flow_output_dict, fusion=cfg.fusion, use_edl=cfg.use_edl, os_head=cfg.os_head)
            decoded_segments, conf_scores, uncertainty, actionness = decode_predictions(loc, prop_loc, priors, conf, prop_conf, unct, prop_unct, act, prop_act, \
                                                                center, offset, sample_fps, cfg.clip_length, cfg.num_classes, score_func=nn.Softmax(dim=-1), use_edl=cfg.use_edl, os_head=cfg.os_head)
            # filtering out clip-level predictions with low confidence
            for cl in range(1, cfg.num_classes):
                segments = filtering(decoded_segments, conf_scores[cl], uncertainty, actionness, cfg.conf_thresh, use_edl=cfg.use_edl, os_head=cfg.os_head)
                if segments is None:
                    continue
                output[cl].append(segments)

        # get final detection results for each video
        result_dict[video_name] = get_video_detections(output, idx_to_class, cfg.num_classes, cfg.top_k, cfg.nms_sigma, use_edl=cfg.use_edl, os_head=cfg.os_head)

    # get the score threshold
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

    output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {'threshold': threshold}}
    with open(output_file, "w") as out:
        json.dump(output_dict, out)
    
    return threshold

    
if __name__ == '__main__':

    cfg = get_basic_config(config)
    
    output_file = os.path.join(cfg.output_path, cfg.json_name)
    if not os.path.exists(output_file):
        threshold = thresholding(cfg, output_file)
    else:
        with open(output_file, 'r') as fobj:
            data = json.load(fobj)
            threshold = data['external_data']['threshold']
        print(f'Thresholding result file already exist at {output_file}!')

    print(f'The threshold is: {threshold:.12f}')