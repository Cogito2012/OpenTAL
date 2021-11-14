import os, json
import copy


def get_video_info(video_info_path, subset='train'):
    with open(video_info_path, 'r') as fobj:
        data = json.load(fobj)
    sub_data = copy.deepcopy(data)
    video_info = {}
    for videoid, v in data['database'].items():
        if subset == v['subset']:
            video_info[videoid] = v
    sub_data['database'] = video_info
    return sub_data
    

# def get_anet_video_info(video_info_path, subset='training'):
#     with open(video_info_path) as json_file:
#         json_data = json.load(json_file)
#     video_info = {}
#     video_list = list(json_data.keys())
#     for video_name in video_list:
#         tmp = json_data[video_name]
#         if tmp['subset'] == subset:
#             video_info[video_name] = tmp
#     return video_info


def exclude_overlapping(anet_infos, overlapping_class_file):
    # read the overlapping class names
    excluded_classes = []
    with open(overlapping_class_file, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            excluded_classes.append(line.strip())
    # filtering out the videos that contain excluded classes
    video_info = {}
    for video_name, info in anet_infos['database'].items():
        exclude = False
        for ann in info['annotations']:
            if ann['label'] in excluded_classes:
                exclude = True
                break
        if not exclude:
            video_info[video_name] = info
    return video_info

    

if __name__ == '__main__':
    splits = ['0', '1', '2']
    # input
    thumos_gt_path = 'datasets/thumos14/annotations/thumos_gt.json'
    anet_gt_path = 'datasets/activitynet/annotations/activity_net_1_3_new.json'
    anet_subset = 'validation'
    overlapping_class_file = 'datasets/activitynet/overlapping_classes_in_thumos.txt'
    # output
    merged_gt_file = 'datasets/thumos14/annotations/thumos_anet_gt.json'

    # get all thumos testing video annotations (210)
    thumos_infos = get_video_info(thumos_gt_path, subset='test')
    merged_gt = copy.deepcopy(thumos_infos)

    num_videos = len(merged_gt['database'])
    print(f'Before merge: {num_videos} videos.')

    # get all anet testing videos and filter out overlapping class
    anet_infos = get_video_info(anet_gt_path, subset='validation')
    anet_infos = exclude_overlapping(anet_infos, overlapping_class_file)
    # merge_all
    merged_gt['database'].update(anet_infos)

    with open(merged_gt_file, 'w') as f:
        json.dump(merged_gt, f)

    num_videos = len(merged_gt['database'])
    print(f'After merge: {num_videos} videos.')