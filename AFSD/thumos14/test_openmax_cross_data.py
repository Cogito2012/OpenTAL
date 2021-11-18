import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
import json
from AFSD.common import videotransforms
from AFSD.common.thumos_dataset import get_video_info, get_video_anno, get_class_index_map
from AFSD.common.config import config
from AFSD.thumos14.BDNet import BDNet
from test_openmax import compute_mav_dist, files_are_ready, weibull_fitting, decode_output, filtering, get_video_detections
from openmax import OpenMax


def get_path(input_path):
    if os.path.lexists(input_path):
        fullpath = os.path.realpath(input_path) if os.path.islink(input_path) else input_path
        real_name = fullpath.split('/')[-1]
        real_full_path = os.path.join(os.path.dirname(input_path), real_name)
    else:
        raise FileNotFoundError
    return real_full_path

        
def get_offsets(sample_count, clip_length, stride):
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
    data = torch.from_numpy(data).cuda()
    return data


def prepare_clip(data, offset, clip_length):
    clip = data[:, offset: offset + clip_length]
    clip = clip.float()
    clip = (clip / 255.0) * 2.0 - 1.0
    if clip.size(1) < clip_length:
        tmp = torch.zeros([clip.size(0), clip_length - clip.size(1),
                            96, 96]).float().cuda()
        clip = torch.cat([clip, tmp], dim=1)
    clip = clip.unsqueeze(0)
    return clip



def prepare_anet_clip(data, offset, clip_length, crop_size):
    clip = data[:, offset: offset + clip_length]
    clip = clip.float()
    if clip.size(1) < clip_length:
        tmp = torch.ones(
            [clip.size(0), clip_length - clip.size(1), crop_size, crop_size]).float().cuda() * 127.5
        clip = torch.cat([clip, tmp], dim=1)
    clip = clip.unsqueeze(0)
    clip = (clip / 255.0) * 2.0 - 1.0
    return clip


def test(cfg, output_file):
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
    with open(output_file, "w") as out:
        json.dump(output_dict, out)


def test_anet(thumos_cfg, anet_cfg, anet_resfile):

    video_list = list(anet_cfg.video_infos.keys())
    videos_in_disk = [filename[:-4] for filename in os.listdir(anet_cfg.mp4_data_path)]
    video_list = list(set(video_list) & set(videos_in_disk))

    # prepare model
    net = BDNet(in_channels=thumos_cfg.input_channels, training=False)
    net.load_state_dict(torch.load(thumos_cfg.checkpoint_path))
    net.eval().cuda()
    center_crop = videotransforms.CenterCrop(thumos_cfg.crop_size)

    result_dict = {}
    for vid, video_name in tqdm(enumerate(video_list), total=len(video_list), desc='ANet Inference'):
        # setup clip offsets
        sample_fps = anet_cfg.video_infos[video_name]['fps']
        duration = anet_cfg.video_infos[video_name]['duration']
        frame_count = anet_cfg.video_infos[video_name]['frame_num']
        offsetlist = get_offsets(frame_count, thumos_cfg.clip_length, thumos_cfg.stride)

        # load data
        data = prepare_data(anet_cfg.mp4_data_path, video_name, center_crop)  # (3, 768, 96, 96)

        output = [[] for cl in range(thumos_cfg.num_classes)]
        output_dict_all = []
        for offset in offsetlist:
            # prepare clip of a video
            clip = prepare_anet_clip(data, offset, thumos_cfg.clip_length, thumos_cfg.crop_size)
            # run inference
            with torch.no_grad():
                output_dict = net(clip, get_feat=True)
            output_dict_all.append((output_dict, offset))

        # post-processing
        output = {'seg': [[] for cl in range(thumos_cfg.num_classes)], 
                  'feat': [[] for cl in range(thumos_cfg.num_classes)], 
                  'prop_feat': [[] for cl in range(thumos_cfg.num_classes)]}
        for (output_dict, offset) in output_dict_all:
            # decode the outputs (late fusion)
            decoded_segments, conf_scores, _, _ = decode_output(output_dict, offset, sample_fps, thumos_cfg, get_feat=True)
            # filtering out clip-level predictions with low confidence
            for cl in range(1, thumos_cfg.num_classes):  # from 1 to K+1 by default
                out = filtering(decoded_segments, conf_scores[cl], thumos_cfg.conf_thresh)
                if out is None:
                    continue
                output['seg'][cl].append(out['seg'])
                output['feat'][cl].append(out['feat'])
                output['prop_feat'][cl].append(out['prop_feat'])
        # get final detection results for each video
        result_dict[video_name] = get_video_detections(output, thumos_cfg)

    output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {}}
    with open(anet_resfile, "w") as out:
        json.dump(output_dict, out)
    
    return output_dict


def exclude_overlapping(anet_out, overlapping_class_file):
    # read the overlapping class names
    excluded_classes = []
    with open(overlapping_class_file, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            excluded_classes.append(line.strip())
    # filtering out the videos that contain excluded classes
    result_dict = {}
    for video_name, preds in anet_out['results'].items():
        video_info = anet_cfg.video_infos['v_' + video_name]
        exclude = False
        for ann in video_info['annotations']:
            if ann['label'] in excluded_classes:
                exclude = True
                break
        if not exclude:
            result_dict[video_name] = preds
    output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {}}
    return output_dict


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
    outpath = config['testing']['output_path']
    split_folder = outpath.split('/')[-1]
    cfg.output_path = os.path.join(os.path.dirname(os.path.dirname(outpath)), config['testing']['exp_tag'], split_folder)
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path)
    cfg.fusion = False

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


def get_anet_cfg():
    def get_anet_video_info(video_info_path, subset='training'):
        with open(video_info_path) as json_file:
            json_data = json.load(json_file)
        video_info = {}
        video_list = list(json_data.keys())
        for video_name in video_list:
            tmp = json_data[video_name]
            if tmp['subset'] == subset:
                video_info[video_name] = tmp
        return video_info

    class cfg: pass
    cfg.clip_length = 768
    cfg.stride = 768
    cfg.crop_size = 96
    cfg.video_infos = get_anet_video_info('datasets/activitynet/annotations/video_info_train_val.json', subset='validation')
    cfg.mp4_data_path = 'datasets/activitynet/train_val_npy_112'
    cfg.overlapping_class_file = 'datasets/activitynet/overlapping_classes_in_thumos.txt'
    return cfg

    
if __name__ == '__main__':

    thumos_cfg = get_basic_config(config)
    anet_cfg = get_anet_cfg()

    mav_dist_dir = os.path.join(thumos_cfg.output_path, 'mav_dist')
    if not files_are_ready(mav_dist_dir, thumos_cfg):
        compute_mav_dist(mav_dist_dir, thumos_cfg)
    # weibull fitting
    weibull_model, weibull_prop_model = weibull_fitting(thumos_cfg, mav_dist_dir)
    thumos_cfg.openmax_layer = OpenMax(weibull_model)
    thumos_cfg.openmax_prop_layer = OpenMax(weibull_prop_model)

    thumos_resfile = os.path.join(thumos_cfg.output_path, 'thumos14_open_rgb.json')
    if not os.path.exists(thumos_resfile):
        thumos_out = test(thumos_cfg, thumos_resfile)
    else:
        print('Thumos14 test results exist! \n%s'%(thumos_resfile))
        with open(thumos_resfile) as json_file:
            thumos_out = json.load(json_file)
        print(f"Number of thumos videos: {len(thumos_out['results'])}\n")
    
    anet_resfile = os.path.join(thumos_cfg.output_path, 'anet_open_rgb.json')
    if not os.path.exists(anet_resfile):
        anet_out = test_anet(thumos_cfg, anet_cfg, anet_resfile)
    else:
        print('ActivityNet1.3 test results exist! \n%s'%(anet_resfile))
        with open(anet_resfile) as json_file:
            anet_out = json.load(json_file)
        print(f"Number of anet videos (before filtering): {len(anet_out['results'])}\n")
    
    # exclude videos from the overlapping classes
    anet_out = exclude_overlapping(anet_out, anet_cfg.overlapping_class_file)
    print(f"Number of anet videos (after filtering): {len(anet_out['results'])}\n")

    # merge the results
    result_dict = thumos_out['results']
    result_dict.update(anet_out['results'])
    print(f"Number of all merged videos: {len(result_dict)}\n")
    
    output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {}}
    with open(os.path.join(thumos_cfg.output_path, thumos_cfg.json_name), "w") as out:
        json.dump(output_dict, out)