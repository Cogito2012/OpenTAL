# This code is originally from the official ActivityNet repo
# https://github.com/activitynet/ActivityNet
# Small modification from ActivityNet Code

import json
from pickle import APPEND
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import copy
from tqdm import tqdm

from .utils_eval import get_blocked_videos
from .utils_eval import interpolated_prec_rec
from .utils_eval import segment_iou
from .utils_eval import save_curve_data
from .utils_eval import open_set_detection_rate, save_curve_osdr_data
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")



class ANETdetection(object):
    GROUND_TRUTH_FIELDS = ['database']
    # GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None, cls_idx_detection=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 ood_threshold=None, 
                 ood_scoring='confidence',
                 subset=['validation'], 
                 openset=False,
                 draw_auc=False,
                 curve_data_path=None,
                 verbose=False, 
                 check_status=False,
                 dataset='thumos14'):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.ood_threshold = ood_threshold
        self.ood_scoring = ood_scoring
        self.openset = openset
        self.draw_auc = draw_auc
        self.curve_data_path = curve_data_path
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        assert dataset in ['thumos14', 'anet', 'thumos_anet']
        self.dataset = dataset
        # Retrieve blocked videos from server.

        if self.check_status:
            self.blocked_videos = get_blocked_videos()
        else:
            self.blocked_videos = list()

        # read the known classes info
        self.activity_index = self.get_activity_index(cls_idx_detection)

        # Import ground truth and predictions.
        self.ground_truth, self.video_lst = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print ('[INIT] Loaded annotations from {} subset.'.format(subset[0]))
            nr_gt = len(self.ground_truth)
            print ('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print ('\tNumber of predictions: {}'.format(nr_pred))
            print ('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))
        if self.openset:
            self.stats = {}

    def get_activity_index(self, class_info_path):
        class_to_idx = {}
        if self.openset:
            class_to_idx['__unknown__'] = 0  # 0 is reserved for unknown in open set
        if self.dataset in ['thumos14', 'thumos_anet']:
            txt = np.loadtxt(class_info_path, dtype=str)
            for idx, l in enumerate(txt):
                class_to_idx[l[1]] = idx + 1  # starting from 1 to K (K=15 for thumos14)
        else:
            with open(class_info_path, 'r') as f:
                for idx, line in enumerate(f.readlines()):
                    class_to_idx[line.strip()] = idx + 1    # starting from 1 to K (K=150 for activitynet)
        return class_to_idx

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data['database'].items():
            # print(v)
            if v['subset'] not in self.subset:
                continue
            if videoid in self.blocked_videos:
                continue

            for ann in v['annotations']:
                video_lst.append(videoid)
                t_start_lst.append(float(ann['segment'][0]))
                t_end_lst.append(float(ann['segment'][1]))
                if self.openset:
                    if ann['label'] in self.activity_index:
                        label_lst.append(self.activity_index[ann['label']])
                    else:
                        label_lst.append(0)  # the unknown
                else:  # closed set
                    assert ann['label'] in self.activity_index, 'Ground truth json contains invalid class: %s'%(ann['label'])
                    label_lst.append(self.activity_index[ann['label']])

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst})
        return ground_truth, video_lst

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst, uncertainty_lst, actness_lst, ood_score_lst = [], [], [], [], []
        for videoid, v in data['results'].items():
            if videoid in self.blocked_videos:
                continue
            if videoid not in self.video_lst:
                continue
            for result in v:
                if result['label'] not in self.activity_index:
                    continue
                # known/unknown classification
                if self.ood_scoring == 'uncertainty':
                    res_score = result['uncertainty']
                    uncertainty_lst.append(result['uncertainty'])
                elif self.ood_scoring == 'confidence':
                    res_score = 1 - result['score']
                elif self.ood_scoring == 'uncertainty_actionness':
                    res_score = result['uncertainty'] * result['actionness']
                    uncertainty_lst.append(result['uncertainty'])
                    actness_lst.append(result['actionness'])
                elif self.ood_scoring == 'a_by_inv_u':
                    res_score = result['actionness'] / (1 - result['uncertainty'] + 1e-6)
                    uncertainty_lst.append(result['uncertainty'])
                    actness_lst.append(result['actionness'])
                elif self.ood_scoring == 'u_by_inv_a':
                    res_score = result['uncertainty'] / (1 - result['actionness'] + 1e-6)
                    uncertainty_lst.append(result['uncertainty'])
                    actness_lst.append(result['actionness'])
                elif self.ood_scoring == 'half_au':
                    res_score = 0.5 * (result['actionness'] + 1) * result['uncertainty']
                    uncertainty_lst.append(result['uncertainty'])
                    actness_lst.append(result['actionness'])
                ood_score_lst.append(res_score)
                if self.openset and self.ood_threshold is not None and res_score < self.ood_threshold:
                    label = self.activity_index['__unknown__']  # reject the unknown
                else:
                    label = self.activity_index[result['label']]
                video_lst.append(videoid)
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                label_lst.append(label)
                score_lst.append(result['score'])
        pred_dict = {'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst,
                                   'ood_score': ood_score_lst}
        if self.ood_scoring in ['uncertainty', 'uncertainty_actionness']:
            pred_dict.update({'uncertainty': uncertainty_lst})
        if self.ood_scoring == 'uncertainty_actionness':
            pred_dict.update({'actionness': actness_lst})
        prediction = pd.DataFrame(pred_dict)
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            if self.verbose:
                print ('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')

        results = Parallel(n_jobs=len(self.activity_index))(
                    delayed(compute_average_precision_detection)(
                        ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                        prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                        tiou_thresholds=self.tiou_thresholds,
                    ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):  # activity_index starting from 1
            ap[:,cidx-1] = results[i]

        # fp = np.zeros((len(self.tiou_thresholds)))
        # tp = np.zeros((len(self.tiou_thresholds)))
        # for i, cidx in enumerate(self.activity_index.values()):
        #     fp += results[i][1]
        #     tp += results[i][2]
        # print(fp, tp)
        return ap

    def wrapper_compute_auc_scores(self):
        pred_scores, pred_labels, gt_labels = self.eval_data
        au_roc, au_pr, roc_data, pr_data = compute_auc_scores(pred_scores, gt_labels, tiou_thresholds=self.tiou_thresholds, vis=self.draw_auc)
        if self.draw_auc:
            save_curve_data(roc_data, pr_data, self.curve_data_path, vis=True)
        return au_roc, au_pr

    
    def wrapper_compute_osdr_scores(self):
        pred_scores, pred_labels, gt_labels = self.eval_data
        osdr, osdr_data = compute_osdr_scores(pred_scores, pred_labels, gt_labels, tiou_thresholds=self.tiou_thresholds, vis=self.draw_auc)
        if self.draw_auc:
            save_curve_osdr_data(osdr_data, self.curve_data_path, vis=True)
        return osdr


    def wrapper_compute_wilderness_impact(self):
        """Computes wilderness impact for each class in the subset.
        """
        assert '__unknown__' in self.activity_index
        activity_index_known = copy.deepcopy(self.activity_index)
        del activity_index_known['__unknown__']

        unique_videos = list(set(self.video_lst))
        wi, self.stats = compute_wilderness_impact(self.ground_truth, self.prediction, unique_videos, activity_index_known,
                                            tiou_thresholds=self.tiou_thresholds)
        return wi


    def pre_evaluate(self):
        unique_videos = list(set(self.video_lst))
        print('For evaluating AUC curves...')
        self.eval_data = split_results_by_gt(self.prediction, self.ground_truth, unique_videos, tiou_thresholds=self.tiou_thresholds)


    def evaluate(self, type='AP'):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """ 
        if type == 'AP':
            self.ap = self.wrapper_compute_average_precision()
            self.mAP = self.ap.mean(axis=1)
            self.average_mAP = self.mAP.mean()
            return self.mAP, self.average_mAP, self.ap
        elif type == 'AUC':
            self.au_roc, self.au_pr = self.wrapper_compute_auc_scores()
            return self.au_roc, self.au_pr
        elif type == 'OSDR':
            self.osdr = self.wrapper_compute_osdr_scores()
            return self.osdr
        elif type == 'WI':
            assert self.openset, 'Wilderness Impact Cannot be Evaluated for Closed Set!'
            self.wi = self.wrapper_compute_wilderness_impact()
            self.mWI = self.wi.mean(axis=1)
            self.average_mWI = self.mWI.mean()
            return self.mWI, self.average_mWI, self.wi
        else:
            raise NotImplementedError


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])


    return ap


def split_results_by_gt(prediction_all, ground_truth_all, video_list, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """ Split predictions into background, known, and unknown actions by ground-truth
    """
    ground_truth_by_vid = ground_truth_all.groupby('video-id')
    prediction_by_vid = prediction_all.groupby('video-id')

    def _get_predictions_with_vid(prediction_by_vid, video_name):
        """Get all predicitons of the given video. Return empty DataFrame if there
        is no predcitions with the given video.
        """
        try:
            return prediction_by_vid.get_group(video_name).reset_index(drop=True)
        except:
            return pd.DataFrame()

    # for each pred label, find the ground truth label
    pred_scores = [{'bg': [], 'known': [], 'unknown': []} for _ in tiou_thresholds]
    pred_labels = [{'bg': [], 'known': [], 'unknown': []} for _ in tiou_thresholds]
    gt_labels = [{'bg': [], 'known': [], 'unknown': []} for _ in tiou_thresholds]
    for video_name in tqdm(video_list, total=len(video_list), desc='Compute AUC', position=0, leave=True):
        ground_truth = ground_truth_by_vid.get_group(video_name).reset_index()
        prediction = _get_predictions_with_vid(prediction_by_vid, video_name)
        if prediction.empty:
            continue
        lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
        for idx, this_pred in prediction.iterrows():
            ood_score = this_pred['ood_score']  # high value indicates unknown class
            label_pred = this_pred['label']  # all predicted classes are known classes without using threshold here!
            tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                                   ground_truth[['t-start', 't-end']].values)
            tiou_sorted_idx = tiou_arr.argsort()[::-1]  # tIoU in a decreasing order
            for tidx, tiou_thr in enumerate(tiou_thresholds):
                for jdx in tiou_sorted_idx:
                    if tiou_arr[jdx] < tiou_thr:  # background segment
                        pred_scores[tidx]['bg'].append(ood_score)
                        pred_labels[tidx]['bg'].append(label_pred)
                        gt_labels[tidx]['bg'].append(-1.0)  # -1: bg
                        break
                    if lock_gt[tidx, jdx] >= 0:
                        continue  # this gt was matched before, continue to select the second largest tIoU match
                    label_gt = int(ground_truth.loc[jdx]['label'])
                    if label_gt == 0: # unknown foreground
                        pred_scores[tidx]['unknown'].append(ood_score)
                        pred_labels[tidx]['unknown'].append(label_pred)
                        gt_labels[tidx]['unknown'].append(label_gt)  # 0: unknown
                    else:  # known foreground
                        pred_scores[tidx]['known'].append(ood_score)
                        pred_labels[tidx]['known'].append(label_pred)
                        gt_labels[tidx]['known'].append(label_gt)  # >0: known
                    lock_gt[tidx, jdx] = idx
                    break
    return pred_scores, pred_labels, gt_labels


def compute_auc_scores(pred_scores, gt_labels, tiou_thresholds=np.linspace(0.5, 0.95, 10), vis=False):
    """ Compute the Area Under the Curves (ROC and PR)
    """
    # compute the AUC of PR and ROC curves between known and unknown
    auc_pr = np.zeros((len(tiou_thresholds),), dtype=np.float32)
    auc_roc = np.zeros((len(tiou_thresholds),), dtype=np.float32)
    roc_data = {'fpr': [], 'tpr': [], 'auc': [], 'tiou': []} if vis else None
    pr_data = {'recall': [], 'precision': [], 'auc': [], 'tiou': []} if vis else None
    for tidx, tiou in enumerate(tiou_thresholds):
        preds = pred_scores[tidx]['known'] + pred_scores[tidx]['unknown']
        labels_cls = gt_labels[tidx]['known'] + gt_labels[tidx]['unknown']
        labels = (1 - np.array(labels_cls).astype(bool).astype(int)).tolist()  # known: 0, unknown: 1
        if len(preds) > 0 and len(labels) > 0:
            auc_pr[tidx] = average_precision_score(labels, preds)  # note that this is interpolated approximation of precision_recall_curve() + auc()
            auc_roc[tidx] = roc_auc_score(labels, preds) if len(list(set(labels))) > 1 else 0  # at least there should be two classes
            if vis:
                # draw AUC_ROC curves
                fpr, tpr, _ = roc_curve(labels, preds, pos_label=1)
                roc_data['fpr'].append(fpr)
                roc_data['tpr'].append(tpr)
                roc_data['auc'].append(auc_roc[tidx])
                roc_data['tiou'].append(tiou)
                # draw AUC_PR curves
                precision, recall, _ = precision_recall_curve(labels, preds, pos_label=1)
                pr_data['precision'].append(precision)
                pr_data['recall'].append(recall)
                pr_data['auc'].append(auc_pr[tidx])
                pr_data['tiou'].append(tiou)
    return auc_roc, auc_pr, roc_data, pr_data



def compute_osdr_scores(pred_scores, pred_labels, gt_labels, tiou_thresholds=np.linspace(0.5, 0.95, 10), vis=False):
    """ Compute the Area Under the CDR-FPR Curve
    """
    osdr = np.zeros((len(tiou_thresholds),), dtype=np.float32)
    osdr_data = {'fpr': [], 'cdr': [], 'osdr': [], 'tiou': []} if vis else None
    for tidx, tiou in enumerate(tiou_thresholds):
        preds = 1 - np.array(pred_scores[tidx]['known'] + pred_scores[tidx]['unknown'])  # confidence. 0: unknown, 1: known
        pred_cls = np.array(pred_labels[tidx]['known'] + pred_labels[tidx]['unknown'])  # integer values ranging from 1-K
        gt_cls = np.array(gt_labels[tidx]['known'] + gt_labels[tidx]['unknown'])  # integer values ranging from 0-K
        if len(preds) > 0:
            osdr[tidx], fpr, cdr = open_set_detection_rate(preds, pred_cls, gt_cls)
            if vis:
                osdr_data['fpr'].append(fpr)
                osdr_data['cdr'].append(cdr)
                osdr_data['osdr'].append(osdr[tidx])
                osdr_data['tiou'].append(tiou)
    return osdr, osdr_data


def compute_wilderness_impact1(ground_truth_all, prediction_all, video_list, known_classes, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """ Compute wilderness impact for each video (WI=Po/Pc < 1)
    """
    wi = np.zeros((len(tiou_thresholds), len(known_classes)))

    # # Initialize true positive and false positive vectors.
    tp_u2u = np.zeros((len(tiou_thresholds), len(prediction_all)))
    tp_k2k = np.zeros((len(tiou_thresholds), len(known_classes), len(prediction_all)))  # TPc in WACV paper
    fp_u2k = np.zeros((len(tiou_thresholds), len(known_classes), len(prediction_all)))  # FPo in WACV paper
    fp_k2k = np.zeros((len(tiou_thresholds), len(known_classes), len(prediction_all)))  # FPc in WACV paper
    fp_k2u = np.zeros((len(tiou_thresholds), len(prediction_all)))
    fp_bg2u = np.zeros((len(tiou_thresholds), len(prediction_all)))
    fp_bg2k = np.zeros((len(tiou_thresholds), len(known_classes), len(prediction_all)))

    ground_truth_by_vid = ground_truth_all.groupby('video-id')
    prediction_by_vid = prediction_all.groupby('video-id')

    def _get_predictions_with_vid(prediction_by_vid, video_name):
        """Get all predicitons of the given video. Return empty DataFrame if there
        is no predcitions with the given video.
        """
        try:
            return prediction_by_vid.get_group(video_name).reset_index(drop=True)
        except:
            return pd.DataFrame()

    # compute the TP, FPo and FPc for each predicted segment.
    vidx_offset = 0
    all_scores, all_max_tious = [], []
    for video_name in tqdm(video_list, total=len(video_list), desc='Compute WI'):
        ground_truth = ground_truth_by_vid.get_group(video_name).reset_index()
        prediction = _get_predictions_with_vid(prediction_by_vid, video_name)

        if prediction.empty:
            vidx_offset += len(prediction)
            all_scores.extend([0] * len(prediction))  # only for confidence score
            all_max_tious.extend([0] * len(prediction))
            continue  # no predictions for this video

        all_scores.extend(prediction['score'].values.tolist())
        lock_gt = np.zeros((len(tiou_thresholds),len(ground_truth)))
        
        for idx, this_pred in prediction.iterrows():
            tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                                ground_truth[['t-start', 't-end']].values)
            # attach each prediction with the gt that has maximum tIoU
            max_iou = tiou_arr.max()
            max_jdx = tiou_arr.argmax()
            all_max_tious.append(max_iou)

            label_pred = this_pred['label']
            label_gt = int(ground_truth.loc[max_jdx]['label'])
            for tidx, tiou_thr in enumerate(tiou_thresholds):
                if max_iou > tiou_thr:
                    if label_pred == label_gt and lock_gt[tidx, max_jdx] == 0:
                        if label_gt == 0:
                            tp_u2u[tidx, vidx_offset + idx] = 1  # true positive (u2u), not used by WI by default
                        else:
                            tp_k2k[tidx, label_pred-1, vidx_offset + idx] = 1  # true positive (k2k)
                        lock_gt[tidx, max_jdx] = 1  # lock this ground truth
                    else:
                        if label_gt == 0: # false positive (u2k)
                            fp_u2k[tidx, label_pred-1, vidx_offset + idx] = 1
                        else:   # false positive (k2k, k2u)
                            if label_pred == 0:
                                fp_k2u[tidx, vidx_offset + idx] = 1
                            else:
                                fp_k2k[tidx, label_pred-1, vidx_offset + idx] = 1
                else: # GT is defined to be background (known), must be FP
                    if label_pred == 0:
                        fp_bg2u[tidx, vidx_offset + idx] = 1
                    else:
                        fp_bg2k[tidx, label_pred-1, vidx_offset + idx] = 1
        # move the offset 
        vidx_offset += len(prediction)

    stats = {'tp_k2k': tp_k2k, 'tp_u2u': tp_u2u, 'fp_k2k': fp_k2k, 'fp_k2u': fp_k2u, 'fp_u2k': fp_u2k, 'fp_bg2k': fp_bg2k, 'fp_bg2u': fp_bg2u,
             'scores': all_scores, 'max_tious': all_max_tious}
    
    # Here we assume the background detections (small tIoU) are from the background class, which is a known class
    fp_k2u += fp_bg2u
    fp_k2k += fp_bg2k

    tp_k2k_sum = np.sum(tp_k2k, axis=-1).astype(np.float)
    fp_u2k_sum = np.sum(fp_u2k, axis=-1).astype(np.float)
    fp_k2k_sum = np.sum(fp_k2k, axis=-1).astype(np.float)
    wi = fp_u2k_sum / (tp_k2k_sum + fp_k2k_sum + 1e-6)

    return wi, stats


def compute_wilderness_impact(ground_truth_all, prediction_all, video_list, known_classes, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """ Compute wilderness impact for each video (WI=Po/Pc < 1)
    """
    wi = np.zeros((len(tiou_thresholds), len(known_classes)))

    # # Initialize true positive and false positive vectors.
    tp_u2u = np.zeros((len(tiou_thresholds), len(prediction_all)))
    tp_k2k = np.zeros((len(tiou_thresholds), len(known_classes), len(prediction_all)))  # TPc in WACV paper
    fp_u2k = np.zeros((len(tiou_thresholds), len(known_classes), len(prediction_all)))  # FPo in WACV paper
    fp_k2k = np.zeros((len(tiou_thresholds), len(known_classes), len(prediction_all)))  # FPc in WACV paper
    fp_k2u = np.zeros((len(tiou_thresholds), len(prediction_all)))
    fp_bg2u = np.zeros((len(tiou_thresholds), len(prediction_all)))
    fp_bg2k = np.zeros((len(tiou_thresholds), len(known_classes), len(prediction_all)))

    ground_truth_by_vid = ground_truth_all.groupby('video-id')
    prediction_by_vid = prediction_all.groupby('video-id')

    def _get_predictions_with_vid(prediction_by_vid, video_name):
        """Get all predicitons of the given video. Return empty DataFrame if there
        is no predcitions with the given video.
        """
        try:
            return prediction_by_vid.get_group(video_name).reset_index(drop=True)
        except:
            return pd.DataFrame()

    # compute the TP, FPo and FPc for each predicted segment.
    vidx_offset = 0
    all_ood_scores, all_scores, all_max_tious = [], [], []
    num_gt = np.zeros((len(known_classes) + 1,), dtype=np.float32)
    for video_name in tqdm(video_list, total=len(video_list), desc='Compute WI'):
        ground_truth = ground_truth_by_vid.get_group(video_name).reset_index()
        prediction = _get_predictions_with_vid(prediction_by_vid, video_name)

        for _, gt in ground_truth.iterrows():
            num_gt[gt['label']] += 1  # keep track of the number of GTs

        if prediction.empty:
            vidx_offset += len(prediction)
            all_ood_scores.extend([0] * len(prediction)) 
            all_scores.extend([0] * len(prediction))  # only for confidence score
            all_max_tious.extend([0] * len(prediction))
            continue  # no predictions for this video

        all_scores.extend(prediction['score'].values.tolist())
        all_ood_scores.extend(prediction['ood_score'].values.tolist())
        lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
        
        for idx, this_pred in prediction.iterrows():
            tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                                ground_truth[['t-start', 't-end']].values)
            tiou_sorted_idx = tiou_arr.argsort()[::-1]  # tIoU in a decreasing order
            all_max_tious.append(tiou_arr[tiou_sorted_idx[0]])

            label_pred = this_pred['label']
            for tidx, tiou_thr in enumerate(tiou_thresholds):
                for jdx in tiou_sorted_idx:
                    # If tIoU is too small, we assume the prediction is background (assign to bg2u/bg2k)
                    if tiou_arr[jdx] < tiou_thr:
                        if label_pred == 0:
                            fp_bg2u[tidx, vidx_offset + idx] = 1
                        else:
                            fp_bg2k[tidx, label_pred-1, vidx_offset + idx] = 1
                        break
                    
                    # Otherwise, consider if this GT (jdx) is already used by previous prediction
                    if lock_gt[tidx, jdx] >= 0:
                        continue  # continue to select the second largest tIoU match
                    
                    # After the filters above, the GT is not used AND tIoU is large enough.
                    # Further consider 5 classification cases (TP_u2u, TP_k2k, FP_u2k, FP_k2k, FP_k2u)
                    label_gt = int(ground_truth.loc[jdx]['label'])
                    if label_pred == label_gt:  # TP cases
                        if label_gt == 0:
                            tp_u2u[tidx, vidx_offset + idx] = 1  # true positive (u2u), not used by WI by default
                        else:
                            tp_k2k[tidx, label_pred-1, vidx_offset + idx] = 1  # true positive (k2k)
                        lock_gt[tidx, jdx] = idx  # lock this ground truth after TP assignment
                    else:
                        if label_gt == 0: # false positive (u2k)
                            fp_u2k[tidx, label_pred-1, vidx_offset + idx] = 1
                        else:   # false positive (k2k, k2u)
                            if label_pred == 0:
                                fp_k2u[tidx, vidx_offset + idx] = 1
                            else:
                                fp_k2k[tidx, label_pred-1, vidx_offset + idx] = 1
                    break
        # move the offset 
        vidx_offset += len(prediction)

    stats = {'tp_k2k': tp_k2k, 'tp_u2u': tp_u2u, 'fp_k2k': fp_k2k, 'fp_k2u': fp_k2u, 'fp_u2k': fp_u2k, 'fp_bg2k': fp_bg2k, 'fp_bg2u': fp_bg2u,
             'ood_scores': all_ood_scores, 'scores': all_scores, 'max_tious': all_max_tious, 'num_gt': num_gt}
    
    # # report the AP for known classes
    # ap = np.zeros((len(known_classes), len(tiou_thresholds)), dtype=np.float32)  # K classes
    # for name, cidx in known_classes.items():
    #     # precision
    #     tp_cumsum = np.cumsum(tp_k2k[:, cidx-1, :], axis=1).astype(np.float)
    #     fp_cumsum = np.cumsum(fp_k2k[:, cidx-1, :], axis=1).astype(np.float)
    #     recall_cumsum = tp_cumsum / num_gt[cidx]
    #     # recall
    #     precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)
    #     for tidx in range(len(tiou_thresholds)):
    #         ap[cidx-1, tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])
    # for tidx, tiou in enumerate(tiou_thresholds):
    #     print(f'tiou={tiou}, AP={ap[:, tidx].mean()}')

    # Here we assume the background detections (small tIoU) are from the background class, which is a known class
    fp_k2u += fp_bg2u
    fp_k2k += fp_bg2k

    # impact on recall ratio
    tp_u2u_cumsum = np.cumsum(tp_u2u, axis=-1).astype(np.float)  # T x N
    recall_ratio_cumsum = num_gt[1:].sum() / ( num_gt[1:].sum() + num_gt[0] - tp_u2u_cumsum)  # T x N
    # impact on precision ratio
    tp_k2k_cumsum = np.cumsum(tp_k2k, axis=-1).astype(np.float)  # T x K x N
    fp_u2k_cumsum = np.cumsum(fp_u2k, axis=-1).astype(np.float)  # T x K x N
    fp_k2k_cumsum = np.cumsum(fp_k2k, axis=-1).astype(np.float)  # T x K x N
    precision_ratio_cumsum = (tp_k2k_cumsum + fp_k2k_cumsum) / (tp_k2k_cumsum + fp_k2k_cumsum + fp_u2k_cumsum + 1e-6)

    for tidx in range(len(tiou_thresholds)):
        for cidx in range(len(known_classes)):
            wi[tidx, cidx] = interpolated_prec_rec(precision_ratio_cumsum[tidx, cidx, :], recall_ratio_cumsum[tidx, :])

    return wi, stats