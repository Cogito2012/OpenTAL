import torch
import torch.nn as nn
import os
import numpy as np
import tqdm
import json
from AFSD.common import videotransforms
from AFSD.common.thumos_dataset import get_video_info, get_class_index_map
from AFSD.thumos14.BDNet import BDNet
from AFSD.common.segment_utils import softnms_v2
from AFSD.common.config import config


def build_model(fusion=False):
    net, flow_net = None, None
    if fusion:
        rgb_net = BDNet(in_channels=3, training=False)
        flow_net = BDNet(in_channels=2, training=False)
        rgb_checkpoint_path = config['testing'].get('rgb_checkpoint_path',
                                            './models/thumos14/checkpoint-15.ckpt')
        flow_checkpoint_path = config['testing'].get('flow_checkpoint_path',
                                             './models/thumos14_flow/checkpoint-16.ckpt')
        rgb_net.load_state_dict(torch.load(rgb_checkpoint_path))
        flow_net.load_state_dict(torch.load(flow_checkpoint_path))
        rgb_net.eval().cuda()
        flow_net.eval().cuda()
        net = rgb_net
    else:
        net = BDNet(in_channels=config['model']['in_channels'],
                    training=False)
        checkpoint_path = config['testing']['checkpoint_path']
        net.load_state_dict(torch.load(checkpoint_path))
        net.eval().cuda()
    return net, flow_net

        
def get_offsets(video_infos, video_name, clip_length, stride):
    sample_count = video_infos[video_name]['sample_count']
    if sample_count < clip_length:
        offsetlist = [0]
    else:
        offsetlist = list(range(0, sample_count - clip_length + 1, stride))
        if (sample_count - clip_length) % stride:
            offsetlist += [sample_count - clip_length]
    return offsetlist


def prepare_data(data_path, video_name, centor_crop):
    data = np.load(os.path.join(data_path, video_name + '.npy'))
    data = np.transpose(data, [3, 0, 1, 2])
    data = centor_crop(data)
    data = torch.from_numpy(data)
    return data


def prepare_clip(data, offset, clip_length):
    clip = data[:, offset: offset + clip_length]
    clip = clip.float()
    clip = (clip / 255.0) * 2.0 - 1.0
    if clip.size(1) < clip_length:
        tmp = torch.zeros([clip.size(0), clip_length - clip.size(1),
                            96, 96]).float()
        clip = torch.cat([clip, tmp], dim=1)
    clip = clip.unsqueeze(0).cuda()
    return clip


def parse_output(output_dict, flow_output_dict=None, fusion=False):
    loc, conf, priors = output_dict['loc'][0], output_dict['conf'][0], output_dict['priors']
    prop_loc, prop_conf = output_dict['prop_loc'][0], output_dict['prop_conf'][0]
    center = output_dict['center'][0]
    if fusion:
        flow_loc, flow_conf, priors = flow_output_dict['loc'][0], flow_output_dict['conf'][0], flow_output_dict['priors']
        flow_prop_loc, flow_prop_conf = flow_output_dict['prop_loc'][0], flow_output_dict['prop_conf'][0]
        flow_center = flow_output_dict['center'][0]
        # fusion by average
        loc = (loc + flow_loc) / 2.0
        prop_loc = (prop_loc + flow_prop_loc) / 2.0
        conf = (conf + flow_conf) / 2.0
        prop_conf = (prop_conf + flow_prop_conf) / 2.0
        center = (center + flow_center) / 2.0
    return loc, conf, prop_loc, prop_conf, center, priors


def decode_predictions(loc, prop_loc, priors, conf, prop_conf, center, \
                        offset, sample_fps, clip_length, num_classes, score_func=nn.Softmax(dim=-1)):
    pre_loc_w = loc[:, :1] + loc[:, 1:]
    loc = 0.5 * pre_loc_w * prop_loc + loc
    segments = torch.cat(
        [priors[:, :1] * clip_length - loc[:, :1],
            priors[:, :1] * clip_length + loc[:, 1:]], dim=-1)
    segments.clamp_(min=0, max=clip_length)
    decoded_segments = (segments + offset) / sample_fps

    conf = score_func(conf)
    prop_conf = score_func(prop_conf)
    center = center.sigmoid()

    conf = (conf + prop_conf) / 2.0
    conf = conf * center
    conf = conf.view(-1, num_classes).transpose(1, 0)
    conf_scores = conf.clone()
    return decoded_segments, conf_scores


def filtering(decoded_segments, conf_score_cls, conf_thresh):
    c_mask = conf_score_cls > conf_thresh
    scores = conf_score_cls[c_mask]
    if scores.size(0) == 0:
        return None
    l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
    segments = decoded_segments[l_mask].view(-1, 2)
    # decode to original time
    segments = torch.cat([segments, scores.unsqueeze(1)], -1)
    return segments


def get_video_detections(output, idx_to_class, num_classes, top_k, nms_sigma):
    res = torch.zeros(num_classes, top_k, 3)
    sum_count = 0
    for cl in range(1, num_classes):
        if len(output[cl]) == 0:
            continue
        tmp = torch.cat(output[cl], 0)
        tmp, count = softnms_v2(tmp, sigma=nms_sigma, top_k=top_k)
        res[cl, :count] = tmp
        sum_count += count

    sum_count = min(sum_count, top_k)
    flt = res.contiguous().view(-1, 3)
    flt = flt.view(num_classes, -1, 3)
    proposal_list = []
    for cl in range(1, num_classes):
        class_name = idx_to_class[cl]
        tmp = flt[cl].contiguous()
        tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, 3)
        if tmp.size(0) == 0:
            continue
        tmp = tmp.detach().cpu().numpy()
        for i in range(tmp.shape[0]):
            tmp_proposal = {}
            tmp_proposal['label'] = class_name
            tmp_proposal['score'] = float(tmp[i, 2])
            tmp_proposal['segment'] = [float(tmp[i, 0]),
                                        float(tmp[i, 1])]
            proposal_list.append(tmp_proposal)
    return proposal_list


def test(cfg):
    # get data
    video_infos = get_video_info(config['dataset']['testing']['video_info_path'])
    _, idx_to_class = get_class_index_map(config['dataset']['class_info_path'])
    npy_data_path = cfg.rgb_data_path if cfg.fusion else config['dataset']['testing']['video_data_path']

    # prepare model
    net, flow_net = build_model(fusion=cfg.fusion)

    centor_crop = videotransforms.CenterCrop(config['dataset']['testing']['crop_size'])
    result_dict = {}
    for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
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

            loc, conf, prop_loc, prop_conf, center, priors = parse_output(output_dict, flow_output_dict, fusion=cfg.fusion)
            decoded_segments, conf_scores = decode_predictions(loc, prop_loc, priors, conf, prop_conf, \
                                                                center, offset, sample_fps, cfg.clip_length, cfg.num_classes, score_func=nn.Softmax(dim=-1))
            # filtering out clip-level predictions with low confidence
            for cl in range(1, cfg.num_classes):
                segments = filtering(decoded_segments, conf_scores[cl], cfg.conf_thresh)
                if segments is None:
                    continue
                output[cl].append(segments)

        # get final detection results for each video
        result_dict[video_name] = get_video_detections(output, idx_to_class, cfg.num_classes, cfg.top_k, cfg.nms_sigma)

    output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {}}
    with open(os.path.join(cfg.output_path, cfg.json_name), "w") as out:
        json.dump(output_dict, out)


def get_basic_config(config):
    class cfg: pass
    cfg.num_classes = config['dataset']['num_classes']
    cfg.conf_thresh = config['testing']['conf_thresh']
    cfg.top_k = config['testing']['top_k']
    cfg.nms_thresh = config['testing']['nms_thresh']
    cfg.nms_sigma = config['testing']['nms_sigma']
    cfg.clip_length = config['dataset']['testing']['clip_length']
    cfg.stride = config['dataset']['testing']['clip_stride']

    cfg.json_name = config['testing']['output_json']
    cfg.output_path = config['testing']['output_path']
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path)
    cfg.fusion = config['testing']['fusion']

    # getting path for fusion
    cfg.rgb_data_path = config['testing'].get('rgb_data_path',
                                        './datasets/thumos14/test_npy/')
    cfg.flow_data_path = config['testing'].get('flow_data_path',
                                        './datasets/thumos14/test_flow_npy/')
    return cfg

    
if __name__ == '__main__':

    cfg = get_basic_config(config)

    test(cfg)