import os, shutil
import numpy as np
import pprint
import pandas as pd
import json
import copy

def get_class_names(class_info_path):
    all_class_names = []
    with open(class_info_path, 'r') as f:
        for line in f.readlines():
            all_class_names.append(line.strip())  # each line is a class name
    return all_class_names


def write_to_txt(filename, class_names):
    with open(filename, 'w') as f:
        for name in class_names:
            f.writelines(name + '\n')


def get_video_info(video_info_path):
    # load json data
    with open(video_info_path) as json_file:
        json_data = json.load(json_file)
    return json_data


def split_classes_random(all_classes, unknown_ratio=0.25):
    num_cls = len(all_classes)
    # randomly select 1/4 categories (=50) as the unknown
    unknown = np.random.choice(num_cls, size=int(num_cls * unknown_ratio), replace=False)
    classes_unknown = [all_classes[idx] for idx in unknown]
    # the rest 3/4 classes are known classes
    classes_known = list(set(class_names_all) - set(classes_unknown))
    return classes_known, classes_unknown


def get_class_ids(classes_known, classes_unknown):
    # coding the class ID, starting from the known classes
    class_to_id = {}
    for idx, name in enumerate(classes_known + classes_unknown):
        class_to_id[name] = idx + 1  # known + unknown class IDs: [1, 2, ..., K + U]
    return class_to_id


def filtering_unknown(result_file, video_info, known_classes, class_to_id):
    filtered_video_info = {}
    for video_name in list(video_info.keys())[:]:
        this_video_info = copy.deepcopy(video_info[video_name])
        subset = this_video_info['subset']
        annotations = this_video_info['annotations']  # a list of dict()
        if subset == 'training':  # need to be filtering by known classes
            # get the filtered anno list
            annos_filtered = []
            for anno in annotations:
                if anno['label'] in known_classes:
                    anno['label_id'] = class_to_id[anno['label']]  # update the class_id
                    annos_filtered.append(anno)
            # update video-level anno
            if len(annos_filtered) > 0: # not empty after filtering
                this_video_info['annotations'] = annos_filtered
                filtered_video_info[video_name] = this_video_info
            else:
                continue # if empty, discard the annotation of this video
        else:  
            # for validation, only need to update class_id
            annos_new = []
            for i, anno in enumerate(annotations):
                anno['label_id'] = class_to_id[anno['label']]  # update the class_id
                annos_new.append(anno)
            this_video_info['annotations'] = annos_new
            filtered_video_info[video_name] = this_video_info
    # save
    with open(result_file, 'w') as f:
        json.dump(filtered_video_info, f)
    return filtered_video_info


def get_anno_stats(video_info):
    num_classes, num_actions = [], []
    for video_name in list(video_info.keys())[:]:
        annotations = video_info[video_name]['annotations']
        num_actions.append(len(annotations))
        class_set = set()
        for anno in annotations:
            class_set.add(anno['label'])
        num_classes.append(len(class_set))
    return num_classes, num_actions

    

if __name__ == '__main__':
    np.random.seed(123)
    
    result_anno_path = 'activitynet/annotations_open'
    os.makedirs(result_anno_path, exist_ok=True)

    # copy the class mapping file
    anno_path = 'activitynet/annotations'
    class_info_file = os.path.join(anno_path, 'action_name.txt')
    assert os.path.exists(class_info_file), f"File does not exist! {class_info_file}"
    shutil.copyfile(class_info_file, os.path.join(result_anno_path, 'action_name.txt'))
    
    # read the class mapping file
    class_names_all = get_class_names(class_info_file)
    num_cls = len(class_names_all)
    print('All categories: (%d)'%(num_cls))

    # read the big json annotation file
    video_info_file = os.path.join(anno_path, 'video_info_train_val.json')
    video_info_all = get_video_info(video_info_file)
    num_classes, num_actions = get_anno_stats(video_info_all)
    print(f'All videos: ({len(video_info_all)}), \
            classes_per_vid: (max={np.max(num_classes)}, mean={np.mean(num_classes):.2f}, min={np.min(num_classes)}), \
            actions_per_vid: (max={np.max(num_actions)}, mean={np.mean(num_actions):.2f}, min={np.min(num_actions)})')

    for i in range(5):  # we provide 5 random splits between known and unknown categories
        split_path = os.path.join(result_anno_path, f'split_{i}')
        os.makedirs(split_path, exist_ok=True)

        # randomly select 1/4 categories (=50) as the unknown
        classes_known, classes_unknown = split_classes_random(class_names_all, unknown_ratio=1/4)
        class_to_id = get_class_ids(classes_known, classes_unknown)  # Class IDs starting from 1 to K+U, topK are known classes
        write_to_txt(os.path.join(split_path, 'action_all.txt'), classes_known + classes_unknown)
        write_to_txt(os.path.join(split_path, 'action_known.txt'), classes_known)

        # filter out the unknown classes in train set
        filtered_video_info = filtering_unknown(os.path.join(split_path, 'video_info_trainval_openset.json'), video_info_all, classes_known, class_to_id)
        num_classes, num_actions = get_anno_stats(video_info_all)
        
        print(f'Videos in split {i}: ({len(filtered_video_info)}),\
            classes_per_vid: (max={np.max(num_classes)}, mean={np.mean(num_classes):.2f}, min={np.min(num_classes)}), \
            actions_per_vid: (max={np.max(num_actions)}, mean={np.mean(num_actions):.2f}, min={np.min(num_actions)})')
