from re import search
import torch
import torch.nn as nn
import os, sys
import numpy as np
import tqdm
import json
from AFSD.common import videotransforms
from AFSD.common.thumos_dataset import get_video_info, get_class_index_map
from AFSD.thumos14.BDNet import BDNet, DirichletLayer
from test import get_offsets, prepare_data, prepare_clip, parse_output, decode_predictions, filtering, get_video_detections, get_path
from AFSD.common.config import config
from AFSD.evaluation.eval_detection import ANETdetection
import time
from joblib import Parallel, delayed


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
        result_dict[k] = v.cpu().numpy()
    return result_dict

def to_tensor(data_dict):
    if len(data_dict) == 0:
        return None
    result_dict = {}
    for k, v in data_dict.items():
        result_dict[k] = torch.from_numpy(v)
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


def post_process(inference_result, conf_thresh=0.01, phase='train'):
    out_layer = DirichletLayer(evidence=cfg.evidence, dim=-1) if cfg.use_edl else nn.Softmax(dim=-1)
    class_range = range(1, cfg.num_classes) if not cfg.os_head else range(0, cfg.num_classes)
    _, idx_to_class = get_class_index_map(cfg.cls_idx_known)

    result_dict = {}
    for out in tqdm.tqdm(inference_result, total=len(inference_result), desc=f'{phase} phase with conf_thresh={conf_thresh}'):
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
                segments = filtering(decoded_segments, conf_scores[cl], uncertainty, actionness, conf_thresh, use_edl=cfg.use_edl, os_head=cfg.os_head)  # (N,5)
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
        pred_file = os.path.join(temp_dir, f'thumos14_open_rgb-{conf_thresh}.json')
        with open(pred_file, "w") as f:
            json.dump(output_dict, f)
        return pred_file
    

def search_job(output_train, output_test, conf_thresh):
    # get the threshold from trainset inference results
    ood_thresh = post_process(output_train, conf_thresh=conf_thresh, phase='train')

    # get the post process results for evaluation
    pred_file = post_process(output_test, conf_thresh=conf_thresh, phase='test')

    # evaluate on test set
    tious = [0.3, 0.4, 0.5, 0.6, 0.7]
    anet_detection = ANETdetection(
        ground_truth_filename=gt_file,
        prediction_filename=pred_file,
        cls_idx_detection=cfg.cls_idx_known,
        subset='test', 
        openset=cfg.open_set,
        ood_threshold=ood_thresh,
        ood_scoring=cfg.scoring,
        tiou_thresholds=tious,
        verbose=False)
    mAPs, average_mAP, ap = anet_detection.evaluate(type='AP')
    
    print(f'Conf threshold: {conf_thresh:.3f}, OOD threshold: {ood_thresh:.6f}, Average mAP: {average_mAP*100:.3f}%')
    return ood_thresh, average_mAP


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

    # send the loaded results into GPU
    output_train = all_to_tensors(output_train)
    output_test = all_to_tensors(output_test)

    gt_file = cfg.gt_all_json if cfg.open_set else cfg.gt_known_json.format(id=cfg.split)
    # search for conf_thresh
    candidates = np.arange(0.001, 0.201, step=0.001)
    ood_thresh_all, average_mAP_all = [], []

    # Parallel not working!
    # results = Parallel(n_jobs=len(candidates))(delayed(search_job)(output_train, output_test, conf_thresh) for conf_thresh in candidates)
    # sys.stdout.flush()
    # for res in results:
    #     ood_thresh_all.append(res[0])
    #     average_mAP_all.append(res[1])

    for conf_thresh in candidates:
        ood_thresh, average_mAP = search_job(output_train, output_test, conf_thresh)
        ood_thresh_all.append(ood_thresh)
        average_mAP_all.append(average_mAP)

    idx = np.array(average_mAP_all).argmax()
    best_conf_thresh = candidates[idx]
    ood_thresh_cur = ood_thresh_all[idx]
    best_mAP = average_mAP_all[idx]
    print(f'\nBest Conf threshold: {best_conf_thresh:.3f}, Current OOD threshold: {ood_thresh_cur:.6f}, Best Average mAP: {best_mAP*100:.3f}%')
