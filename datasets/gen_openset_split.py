import os, shutil
import numpy as np
import pprint
import pandas as pd
import json
import copy

def get_class_index_map(class_info_path):
    txt = np.loadtxt(class_info_path, dtype=str)
    originidx_to_idx = {}
    idx_to_class = {}
    for idx, l in enumerate(txt):
        originidx_to_idx[int(l[0])] = idx + 1
        idx_to_class[idx + 1] = l[1]
    return originidx_to_idx, idx_to_class

def write_to_txt(filename, idx_to_class, originidx_to_idx):
    with open(filename, 'w') as f:
        for ori_idx, idx in originidx_to_idx.items():
            f.writelines(f'{ori_idx} {idx_to_class[idx]}\n')

def csv_filtering(result_csv, anno_file, filtered_class):
    assert os.path.exists(anno_file), f'File does not exist! {anno_file}'
    df_anno = pd.read_csv(anno_file)
    df_anno = df_anno.dropna(how='all')
    df_anno.drop(df_anno[df_anno['type'].isin(filtered_class)].index, inplace=True)
    df_anno.to_csv(result_csv, index=False)

def json_filtering(result_json, gt_file, filtered_class):
    assert os.path.exists(gt_file), f'File does not exist! {gt_file}'
    with open(gt_file, 'r') as fobj:
        data = json.load(fobj)
    new_gt = copy.deepcopy(data)
    for videoid, v in data['database'].items():
        # iterate on each segments
        anno_list = []
        for ann in v['annotations']:
            if ann['label'] not in filtered_class:
                anno_list.append(ann)
        if len(anno_list) > 0:  # not empty after filtering
            v_new = copy.deepcopy(v)
            v_new['annotations'] = anno_list
            new_gt['database'][videoid] = v_new
        else:
            # if empty, delete annotation of this video
            new_gt['database'].pop(videoid)
    # save to json file
    with open(result_json, 'w') as f:
        json.dump(new_gt, f)
    

if __name__ == '__main__':
    np.random.seed(123)
    
    result_anno_path = '../datasets/thumos14/annotations_open'
    os.makedirs(result_anno_path, exist_ok=True)

    # copy the class mapping file
    anno_path = '../datasets/thumos14/annotations'
    class_info_file = os.path.join(anno_path, 'Class_Index_Detection.txt')
    assert os.path.exists(class_info_file), f"File does not exist! {class_info_file}"
    shutil.copyfile(class_info_file, os.path.join(result_anno_path, 'Class_Index_Detection.txt'))
    
    # read the class mapping file
    originidx_to_idx, idx_to_class = get_class_index_map(class_info_file)
    pp = pprint.PrettyPrinter(indent=4)
    print('All categories: (%d)'%(len(idx_to_class)))
    pp.pprint(idx_to_class)

    for i in range(5):  # we provide 5 random splits between known and unknown categories
        split_path = os.path.join(result_anno_path, f'split_{i}')
        os.makedirs(split_path, exist_ok=True)

        # randomly select 5 categories as the unknown
        unknown = np.random.choice(len(idx_to_class), size=5, replace=False)
        idx_to_unknown = dict(filter(lambda elem: elem[0] in unknown, idx_to_class.items()))
        originidx_to_idx_unknown = dict(filter(lambda elem: elem[1] in unknown, originidx_to_idx.items()))
        write_to_txt(os.path.join(split_path, 'Class_Index_Unknown.txt'), idx_to_unknown, originidx_to_idx_unknown)

        # the rest 15 items are known classes
        idx_to_known = dict(filter(lambda elem: elem[0] not in unknown, idx_to_class.items()))
        originidx_to_idx_known = dict(filter(lambda elem: elem[1] not in unknown, originidx_to_idx.items()))
        write_to_txt(os.path.join(split_path, 'Class_Index_Known.txt'), idx_to_known, originidx_to_idx_known)

        # filter out the unknown classes in val set
        csv_filtering(os.path.join(split_path, 'val_Annotation_known.csv'), 
                      os.path.join(anno_path, 'val_Annotation_ours.csv'), list(idx_to_unknown.values()))
        csv_filtering(os.path.join(split_path, 'val_Annotation_unknown.csv'), 
                      os.path.join(anno_path, 'val_Annotation_ours.csv'), list(idx_to_known.values()))
        # filter out the unknown classes in test set
        csv_filtering(os.path.join(split_path, 'test_Annotation_known.csv'), 
                      os.path.join(anno_path, 'test_Annotation_ours.csv'), list(idx_to_unknown.values()))
        csv_filtering(os.path.join(split_path, 'test_Annotation_unknown.csv'), 
                      os.path.join(anno_path, 'test_Annotation_ours.csv'), list(idx_to_known.values()))
        
        # filter out JSON gt file for known 
        json_filtering(os.path.join(split_path, 'known_gt.json'),
                       os.path.join(anno_path, 'thumos_gt.json'), list(idx_to_unknown.values()))
        # filter out JSON gt file for unknown
        json_filtering(os.path.join(split_path, 'unknown_gt.json'),
                       os.path.join(anno_path, 'thumos_gt.json'), list(idx_to_known.values()))
        
    # copy video info files
    video_info_file = os.path.join(anno_path, 'val_video_info.csv')
    assert os.path.exists(video_info_file), f'File does not exist! {video_info_file}'
    shutil.copyfile(video_info_file, os.path.join(result_anno_path, 'val_video_info.csv'))

    video_info_file = os.path.join(anno_path, 'test_video_info.csv')
    assert os.path.exists(video_info_file), f'File does not exist! {video_info_file}'
    shutil.copyfile(video_info_file, os.path.join(result_anno_path, 'test_video_info.csv'))

    open_anno_file = os.path.join(anno_path, 'test_Annotation_ours.csv')
    assert os.path.exists(open_anno_file), f'File does not exist! {open_anno_file}'
    shutil.copyfile(open_anno_file, os.path.join(result_anno_path, 'test_Annotation_open.csv'))
    