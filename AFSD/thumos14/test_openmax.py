import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import numpy as np
import tqdm
import json
from AFSD.common import videotransforms
from AFSD.common.thumos_dataset import get_video_info, get_video_anno, get_class_index_map, split_videos, load_video_data, annos_transform
from AFSD.thumos14.BDNet import BDNet
from test import prepare_data, prepare_clip, get_path, get_offsets
from AFSD.common.config import config
from AFSD.common.segment_utils import softnms_v2
try:
    sys.path.insert(0,'experiments/openmax/libMR')
    import libmr
except ImportError:
    print("LibMR not installed or libmr.so not found")
    print("Install libmr: cd libMR/; ./compile.sh")
    sys.exit()
from openmax import OpenMax, compute_eucos_dist


def get_basic_config(config):
    class cfg: pass
    cfg.num_classes = config['dataset']['num_classes']
    cfg.conf_thresh = config['testing']['conf_thresh']
    cfg.top_k = config['testing']['top_k']
    cfg.nms_thresh = config['testing']['nms_thresh']
    cfg.nms_sigma = config['testing']['nms_sigma']
    # testing data config
    cfg.clip_length = config['dataset']['testing']['clip_length']
    cfg.stride = config['dataset']['testing']['clip_stride']
    cfg.crop_size = config['dataset']['testing']['crop_size']
    # training data config
    cfg.clip_length_train = config['dataset']['training']['clip_length']
    cfg.stride_train = config['dataset']['training']['clip_stride']
    cfg.crop_size_train = config['dataset']['training']['crop_size']
    cfg.overlap_thresh = config['training']['piou']

    cfg.input_channels = config['model']['in_channels']
    cfg.checkpoint_path = get_path(config['testing']['checkpoint_path'])
    cfg.feat_dim = 512

    cfg.json_name = config['testing']['output_json']
    cfg.output_path = config['testing']['output_path']
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path)
    cfg.fusion = False

    # class_mapping, different for each split
    _, cfg.idx_to_class = get_class_index_map(config['dataset']['class_info_path'])
    # train data
    cfg.video_data_train = config['dataset']['training']['video_data_path']
    cfg.video_info_train = get_video_info(config['dataset']['training']['video_info_path'])
    cfg.video_anno_train = get_video_anno(cfg.video_info_train,
                                       config['dataset']['training']['video_anno_path'],
                                       config['dataset']['class_info_path'])
    # test data
    cfg.video_data_test = config['dataset']['testing']['video_data_path']
    cfg.video_info_test = get_video_info(config['dataset']['testing']['video_info_path'])

    return cfg


def prepare_train_data(video_data, offset, clip_length, center_crop, annos):
    input_data = video_data[:, offset: offset + clip_length]
    c, t, h, w = input_data.shape
    if t < clip_length:
        # padding t to clip_length
        pad_t = clip_length - t
        zero_clip = np.zeros([c, pad_t, h, w], input_data.dtype)
        input_data = np.concatenate([input_data, zero_clip], 1)
    input_data = center_crop(input_data)
    input_data = torch.from_numpy(input_data).float()
    input_data = (input_data / 255.0) * 2.0 - 1.0
    annos = annos_transform(annos, cfg.clip_length_train)
    target = torch.from_numpy(np.stack(annos, 0))
    return input_data, target


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


def get_matched_targets(targets, loc_data, priors, clip_length, tiou_thresh=0.5):
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
        prop_conf[iou < tiou_thresh] = 0
        prop_conf_t[idx] = prop_conf
        prop_w = pre_loc[:, 0] + pre_loc[:, 1]
        prop_loc_t[idx][:, 0] = (loc_t[idx][:, 0] - pre_loc[:, 0]) / (0.5 * prop_w)
        prop_loc_t[idx][:, 1] = (loc_t[idx][:, 1] - pre_loc[:, 1]) / (0.5 * prop_w)
    return loc_t, conf_t, prop_loc_t, prop_conf_t


def decode_output(output_dict, offset, sample_fps, cfg, get_feat=False):
    # get the raw outputs
    loc, conf, priors = output_dict['loc'][0], output_dict['conf'][0], output_dict['priors']
    prop_loc, prop_conf = output_dict['prop_loc'][0], output_dict['prop_conf'][0]
    center = output_dict['center'][0]
    if get_feat:
        feat, prop_feat = output_dict['conf_feat'][0], output_dict['prop_conf_feat'][0]

    # late fusion
    pre_loc_w = loc[:, :1] + loc[:, 1:]
    loc = 0.5 * pre_loc_w * prop_loc + loc
    segments = torch.cat(
        [priors[:, :1] * cfg.clip_length - loc[:, :1],
            priors[:, :1] * cfg.clip_length + loc[:, 1:]], dim=-1)
    segments.clamp_(min=0, max=cfg.clip_length)
    decoded_segments = (segments + offset) / sample_fps

    conf = cfg.openmax_layer(conf[:, 1:], feat) if get_feat else F.softmax(conf, dim=1) 
    prop_conf = cfg.openmax_prop_layer(prop_conf[:, 1:], feat) if get_feat else F.softmax(prop_conf, dim=1)
    center = center.sigmoid()

    conf = (conf + prop_conf) / 2.0
    conf = conf * center
    conf = conf.view(-1, cfg.num_classes).transpose(1, 0)
    conf_scores = conf.clone()

    out = (decoded_segments, conf_scores)
    if get_feat:
        out += (feat, prop_feat)
    return out


def filtering(decoded_segments, conf_score_cls, conf_thresh, feat=None, prop_feat=None):
    c_mask = conf_score_cls > conf_thresh
    scores = conf_score_cls[c_mask]
    if scores.size(0) == 0:
        return None
    # masking segments
    l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
    segments = decoded_segments[l_mask].view(-1, 2)
    segments = torch.cat([segments, scores.unsqueeze(1)], -1)  # (N, 3)
    
    out = {'seg': segments, 'feat': None, 'prop_feat': None}
    if feat is not None and prop_feat is not None:
        feat_filtered = feat[c_mask]
        prop_feat_filtered = prop_feat[c_mask]
        out['feat'] = feat_filtered
        out['prop_feat'] = prop_feat_filtered
    return out


def get_video_detections(output, cfg, get_feat=False):
    res_dim = 3
    res = torch.zeros(cfg.num_classes, cfg.top_k, res_dim)
    if get_feat:
        res_feat = torch.zeros(cfg.num_classes, cfg.top_k, cfg.feat_dim)
        res_prop_feat = torch.zeros(cfg.num_classes, cfg.top_k, cfg.feat_dim)
    sum_count = 0
    for cl in range(1, cfg.num_classes):  # from 1 to K+1 by default
        if len(output['seg'][cl]) == 0:
            continue
        tmp = torch.cat(output['seg'][cl], 0)
        if get_feat:
            tmp, count, nms_mask = softnms_v2(tmp, sigma=cfg.nms_sigma, top_k=cfg.top_k, score_threshold=0.001, get_mask=True)
            feat, prop_feat = torch.cat(output['feat'][cl], 0), torch.cat(output['prop_feat'][cl], 0)
            res_feat[cl, :count] = feat[nms_mask]
            res_prop_feat[cl, :count] = prop_feat[nms_mask]
        else:
            tmp, count = softnms_v2(tmp, sigma=cfg.nms_sigma, top_k=cfg.top_k, score_threshold=0.001)
        res[cl, :count] = tmp
        sum_count += count

    sum_count = min(sum_count, cfg.top_k)
    flt = res.contiguous().view(-1, res_dim)
    flt = flt.view(cfg.num_classes, -1, res_dim)
    if get_feat:
        flt_feat = res_feat.contiguous().view(-1, cfg.feat_dim).view(cfg.num_classes, -1, cfg.feat_dim)
        flt_prop_feat = res_prop_feat.contiguous().view(-1, cfg.feat_dim).view(cfg.num_classes, -1, cfg.feat_dim)
    proposal_list = []
    for cl in range(1, cfg.num_classes):  # from 1 to K+1 by default
        class_name = cfg.idx_to_class[cl]  # the name of K classes
        tmp = flt[cl].contiguous()  # (topK, 3)
        mask = tmp[:, 2] > 0
        tmp = tmp[mask.unsqueeze(-1).expand_as(tmp)].view(-1, res_dim)
        if tmp.size(0) == 0:
            continue
        if get_feat:
            tmp_feat = flt_feat[cl].contiguous()
            tmp_feat = tmp_feat[mask.unsqueeze(-1).expand_as(tmp_feat)].view(-1, cfg.feat_dim)
            tmp_feat = tmp_feat.detach().cpu().numpy()
            tmp_prop_feat = flt_prop_feat[cl].contiguous()
            tmp_prop_feat = tmp_prop_feat[mask.unsqueeze(-1).expand_as(tmp_prop_feat)].view(-1, cfg.feat_dim)
            tmp_prop_feat = tmp_prop_feat.detach().cpu().numpy()
        tmp = tmp.detach().cpu().numpy()
        for i in range(tmp.shape[0]):
            tmp_proposal = {}
            tmp_proposal['label'] = class_name
            tmp_proposal['score'] = float(tmp[i, 2])
            tmp_proposal['segment'] = [float(tmp[i, 0]),
                                        float(tmp[i, 1])]
            if get_feat:
                tmp_proposal['feat'] = tmp_feat[i]
                tmp_proposal['prop_feat'] = tmp_prop_feat[i]
            proposal_list.append(tmp_proposal)
    return proposal_list


def compute_mav_dist(mav_dist_dir, cfg):
    # prepare model
    net = BDNet(in_channels=cfg.input_channels, training=False)
    net.load_state_dict(torch.load(cfg.checkpoint_path))
    net.eval().cuda()
    center_crop = videotransforms.CenterCrop(cfg.crop_size_train)

    # prepare dataset
    data_list, _ = split_videos(cfg.video_info_train, cfg.video_anno_train, clip_length=cfg.clip_length_train, stride=cfg.stride_train)
    train_data_dict = load_video_data(cfg.video_info_train, cfg.video_data_train)  # load the entire THUMOS14 dataset

    mav_features, mav_prop_features = {}, {}
    all_features, all_prop_features = {}, {}
    for cl, name in cfg.idx_to_class.items():
        mav_features[name] = np.zeros((cfg.feat_dim))
        mav_prop_features[name] = np.zeros((cfg.feat_dim))
        all_features[name] = []
        all_prop_features[name] = []
    # video_list = list(cfg.video_info_train.keys())
    count, count_prop = 0, 0
    for sample_info in tqdm.tqdm(data_list, total=len(data_list), ncols=0, desc='Extracting MAV frome Train Set'):
        video_name = sample_info['video_name']
        sample_fps = cfg.video_info_train[video_name]['sample_fps']
        # get and prepare training clip
        video_data = train_data_dict[video_name]  # (C, T, H, W)
        clip, target = prepare_train_data(video_data, sample_info['offset'], cfg.clip_length_train, center_crop, sample_info['annos'])
        clip = clip.unsqueeze(0).cuda()
        target = [target.cuda()]
        # inference
        with torch.no_grad():
            clip_out = net(clip, get_feat=True)
        
        # decode the outputs (late fusion)
        feat, prop_feat = clip_out['conf_feat'][0], clip_out['prop_conf_feat'][0]
        # find the targets
        _, conf_t, _, prop_conf_t = get_matched_targets(target, clip_out['loc'], clip_out['priors'], cfg.clip_length_train, cfg.overlap_thresh)
        # coarse stage
        target_conf = conf_t.view(-1)
        inds_pos = target_conf  > 0
        labels_pos = target_conf[inds_pos]
        if labels_pos.numel() > 0:
            # save features
            feat_pos = feat[inds_pos].cpu().numpy()
            labels_pos = labels_pos.cpu().numpy()
            for cl, feature in zip(labels_pos, feat_pos):
                name = cfg.idx_to_class[cl]
                # online averaging (\bar{x}_{n+1} = \frac{n}{n+1} * \bar{x}_{n}) + \frac{1}{n+1} * x_{n+1}
                mav_features[name] = count / (count + 1) * mav_features[name] + 1.0/(count + 1) * feature
                all_features[name].append(feature)
                count += 1
        # refined stage
        prop_target_conf = prop_conf_t.view(-1)
        inds_pos = prop_target_conf > 0
        labels_pos = prop_target_conf[inds_pos]
        if labels_pos.numel() > 0:
            # save features
            prop_feat_pos = prop_feat[inds_pos].cpu().numpy()
            labels_pos = labels_pos.cpu().numpy()
            for cl, feature in zip(labels_pos, prop_feat_pos):
                name = cfg.idx_to_class[cl]
                # online averaging (\bar{x}_{n+1} = \frac{n}{n+1} * \bar{x}_{n}) + \frac{1}{n+1} * x_{n+1}
                mav_prop_features[name] = count_prop / (count_prop + 1) * mav_prop_features[name] + 1.0/(count_prop + 1) * feature
                all_prop_features[name].append(feature)
                count_prop += 1
    # save class-wise features into disk
    os.makedirs(mav_dist_dir, exist_ok=True)
    for cls_name in list(cfg.idx_to_class.values()):
        # coarse stage: extract MAV features and distances
        feat_list = all_features[cls_name]
        features = np.stack(feat_list, axis=0)  # (N, D)
        mav_train = np.mean(features, axis=0)  # (D,)
        eucos_dist = np.array([compute_eucos_dist(mav_train, feat) for feat in feat_list])  # (N,)
        # refined stage: extract MAV features and distances
        prop_feat_list = all_prop_features[cls_name]
        features = np.stack(prop_feat_list, axis=0)  # (N, D)
        mav_prop_train = np.mean(features, axis=0)  # (D,)
        prop_eucos_dist = np.array([compute_eucos_dist(mav_prop_train, feat) for feat in prop_feat_list])  # (N,)
        # save results
        mav_dist_file = os.path.join(mav_dist_dir, f'{cls_name}.npz')
        np.savez(mav_dist_file, mav=mav_train, dist=eucos_dist, mav_prop=mav_prop_train, dist_prop=prop_eucos_dist)
        


def weibull_fitting(cfg, mav_dist_dir, tailsize=20):
    weibull_model, weibull_prop_model = {}, {}
    for cl, name in cfg.idx_to_class.items():
        # load the mav and dist data
        mav_dist_file = os.path.join(mav_dist_dir, f'{name}.npz')
        data = np.load(mav_dist_file, allow_pickle=True)
        mav, dist = data['mav'], data['dist']
        mav_prop, dist_prop = data['mav_prop'], data['dist_prop']
        
        # weibull fitting (coarse)
        weibull_model[name] = {'mean_vec': mav, 'model': []}
        mr = libmr.MR()
        tailtofit = sorted(dist)[-tailsize:]  # points with top-K largest distances
        mr.fit_high(tailtofit, len(tailtofit))
        weibull_model[name]['model'].append(mr)

        # weibull fitting (refined)
        weibull_prop_model[name] = {'mean_vec': mav_prop, 'model': []}
        mr = libmr.MR()
        tailtofit = sorted(dist_prop)[-tailsize:]  # points with top-K largest distances
        mr.fit_high(tailtofit, len(tailtofit))
        weibull_prop_model[name]['model'].append(mr)

    return weibull_model, weibull_prop_model



def test(cfg):
    # prepare model
    net = BDNet(in_channels=cfg.input_channels, training=False)
    net.load_state_dict(torch.load(cfg.checkpoint_path))
    net.eval().cuda()
    center_crop = videotransforms.CenterCrop(cfg.crop_size)
    
    result_dict = {}
    for video_name in tqdm.tqdm(list(cfg.video_info_test.keys()), ncols=0):
        # get the clip offsets
        offsetlist = get_offsets(cfg.video_info_test, video_name, cfg.clip_length, cfg.stride)
        sample_fps = cfg.video_info_test[video_name]['sample_fps']
        # load data
        data = prepare_data(cfg.video_data_test, video_name, center_crop)

        output = [[] for cl in range(cfg.num_classes)]
        output_dict_all = []
        for offset in offsetlist:
            # prepare clip of a video
            clip = prepare_clip(data, offset, cfg.clip_length)
            # run inference
            with torch.no_grad():
                output_dict = net(clip, get_feat=True)
            output_dict_all.append((output_dict, offset))

        # post-processing
        output = {'seg': [[] for cl in range(cfg.num_classes)], 
                    'feat': [[] for cl in range(cfg.num_classes)], 
                    'prop_feat': [[] for cl in range(cfg.num_classes)]}
        for (output_dict, offset) in output_dict_all:
            # decode the outputs (late fusion)
            decoded_segments, conf_scores, _, _ = decode_output(output_dict, offset, sample_fps, cfg, get_feat=True)
            # filtering out clip-level predictions with low confidence
            for cl in range(1, cfg.num_classes):  # from 1 to K+1 by default
                out = filtering(decoded_segments, conf_scores[cl], cfg.conf_thresh)
                if out is None:
                    continue
                output['seg'][cl].append(out['seg'])
                output['feat'][cl].append(out['feat'])
                output['prop_feat'][cl].append(out['prop_feat'])
        # get final detection results for each video
        result_dict[video_name] = get_video_detections(output, cfg)
    
    output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {}}
    with open(os.path.join(cfg.output_path, cfg.json_name), "w") as out:
        json.dump(output_dict, out)



def files_are_ready(mav_dist_dir, cfg):
    ready = True
    # check the mav_dist files
    for cl, name in cfg.idx_to_class.items():
        mav_dist_file = os.path.join(mav_dist_dir, f'{name}.npz')
        if not os.path.exists(mav_dist_file):
            ready = False
    return ready
    
if __name__ == '__main__':

    cfg = get_basic_config(config)
    
    mav_dist_dir = os.path.join(cfg.output_path, 'mav_dist')
    if not files_are_ready(mav_dist_dir, cfg):
        compute_mav_dist(mav_dist_dir, cfg)
    
    # weibull fitting
    weibull_model, weibull_prop_model = weibull_fitting(cfg, mav_dist_dir)
    cfg.openmax_layer = OpenMax(weibull_model)
    cfg.openmax_prop_layer = OpenMax(weibull_prop_model)

    # run inference test
    test(cfg)