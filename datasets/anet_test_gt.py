import os, json
import copy


def get_class_names(class_info_path):
    all_class_names = []
    with open(class_info_path, 'r') as f:
        for line in f.readlines():
            all_class_names.append(line.strip())  # each line is a class name
    return all_class_names


def get_video_info(video_info_path):
    # load json data
    with open(video_info_path) as json_file:
        json_data = json.load(json_file)
    return json_data


def get_filtered_database(class_file, video_info, subset='validation'):
    assert os.path.exists(class_file), 'File does not exist!\n%s'%(class_file)
    class_names = get_class_names(class_file)

    database = {}
    for videoid, v in video_info['database'].items():
        if v['subset'] != subset:
            continue  # we only need the val subset
        annos_filtered = []
        this_video_info = copy.deepcopy(video_info['database'][videoid])
        for anno in v['annotations']:
            if anno['label'] in class_names:
                annos_filtered.append(anno)
        if len(annos_filtered) > 0: # not empty after filtering
            this_video_info['annotations'] = annos_filtered
            database[videoid] = this_video_info
    result = {'database': database}
    return result


if __name__ == '__main__':
    # read video info
    test_gt_file = 'activitynet/annotations/activity_net_1_3_new.json'
    video_info = get_video_info(test_gt_file)

    # output file
    output_dir = 'activitynet/annotations_open/'

    for i in range(5):
        # read known class names
        known_cls_file = os.path.join(output_dir, f'split_{i}', 'action_known.txt')
        # filtering with the class file
        database_known = get_filtered_database(known_cls_file, video_info, subset='validation')
        # save the gt video info
        output_file = os.path.join(output_dir, f'split_{i}', 'known_val_gt.json')
        with open(output_file, "w") as out:
            json.dump(database_known, out)

        # read all class names
        all_cls_file = os.path.join(output_dir, f'split_{i}', 'action_all.txt')
        # filtering with the class file
        database_all = get_filtered_database(all_cls_file, video_info, subset='validation')
        # save the gt video info
        output_file = os.path.join(output_dir, f'split_{i}', 'all_val_gt.json')
        with open(output_file, "w") as out:
            json.dump(database_all, out)
