"""
Tracker Benchmark Evaluation Module
严格按照ORTrack Python代码实现 (result_test/lib/test/analysis/extract_results.py)
"""
import numpy as np
from typing import Dict, List, Tuple, Optional


def load_txt_results(filepath: str) -> np.ndarray:
    """加载跟踪结果或ground truth的txt文件"""
    try:
        data = np.loadtxt(filepath, delimiter=',')
    except:
        try:
            data = np.loadtxt(filepath, delimiter='\t')
        except:
            data = np.loadtxt(filepath)
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    return data


# =============================================================================
# ORTrack实现 - 严格按照extract_results.py
# =============================================================================

def calc_err_center(pred_bb, anno_bb, normalized=False):
    """
    严格按照ORTrack的calc_err_center实现 (extract_results.py 第16-25行)
    
    def calc_err_center(pred_bb, anno_bb, normalized=False):
        pred_center = pred_bb[:, :2] + 0.5 * (pred_bb[:, 2:] - 1.0)
        anno_center = anno_bb[:, :2] + 0.5 * (anno_bb[:, 2:] - 1.0)
        if normalized:
            pred_center = pred_center / anno_bb[:, 2:]
            anno_center = anno_center / anno_bb[:, 2:]
        err_center = ((pred_center - anno_center)**2).sum(1).sqrt()
        return err_center
    """
    # pred_center = pred_bb[:, :2] + 0.5 * (pred_bb[:, 2:] - 1.0)
    pred_center = np.column_stack([
        pred_bb[:, 0] + 0.5 * (pred_bb[:, 2] - 1.0),
        pred_bb[:, 1] + 0.5 * (pred_bb[:, 3] - 1.0)
    ])
    
    # anno_center = anno_bb[:, :2] + 0.5 * (anno_bb[:, 2:] - 1.0)
    anno_center = np.column_stack([
        anno_bb[:, 0] + 0.5 * (anno_bb[:, 2] - 1.0),
        anno_bb[:, 1] + 0.5 * (anno_bb[:, 3] - 1.0)
    ])
    
    if normalized:
        # pred_center = pred_center / anno_bb[:, 2:]
        # anno_center = anno_center / anno_bb[:, 2:]
        pred_center = pred_center / anno_bb[:, 2:4]
        anno_center = anno_center / anno_bb[:, 2:4]
    
    # err_center = ((pred_center - anno_center)**2).sum(1).sqrt()
    err_center = np.sqrt(((pred_center - anno_center)**2).sum(axis=1))
    return err_center


def calc_iou_overlap(pred_bb, anno_bb):
    """
    严格按照ORTrack的calc_iou_overlap实现 (extract_results.py 第28-37行)
    
    def calc_iou_overlap(pred_bb, anno_bb):
        tl = torch.max(pred_bb[:, :2], anno_bb[:, :2])
        br = torch.min(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
        sz = (br - tl + 1.0).clamp(0)
        intersection = sz.prod(dim=1)
        union = pred_bb[:, 2:].prod(dim=1) + anno_bb[:, 2:].prod(dim=1) - intersection
        return intersection / union
    """
    # tl = torch.max(pred_bb[:, :2], anno_bb[:, :2])
    tl = np.maximum(pred_bb[:, :2], anno_bb[:, :2])
    
    # br = torch.min(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    br = np.minimum(pred_bb[:, :2] + pred_bb[:, 2:4] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:4] - 1.0)
    
    # sz = (br - tl + 1.0).clamp(0)
    sz = np.maximum(br - tl + 1.0, 0)
    
    # intersection = sz.prod(dim=1)
    intersection = sz[:, 0] * sz[:, 1]
    
    # union = pred_bb[:, 2:].prod(dim=1) + anno_bb[:, 2:].prod(dim=1) - intersection
    union = (pred_bb[:, 2] * pred_bb[:, 3]) + (anno_bb[:, 2] * anno_bb[:, 3]) - intersection
    
    # return intersection / union
    # 注意：PyTorch中0/0=nan, x/0=inf，这里保持一致
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = intersection / union
    return iou


def calc_seq_err_robust(pred_bb, anno_bb, dataset, target_visible=None):
    """
    严格按照ORTrack的calc_seq_err_robust实现 (extract_results.py 第40-100行)
    """
    pred_bb = pred_bb.copy().astype(np.float64)
    anno_bb = anno_bb.copy().astype(np.float64)
    
    # 第54-57行: 处理零尺寸预测
    # if (pred_bb[:, 2:] == 0.0).any():
    #     for i in range(1, pred_bb.shape[0]):
    #         if (pred_bb[i, 2:] == 0.0).any() and not torch.isnan(anno_bb[i, :]).any():
    #             pred_bb[i, :] = pred_bb[i-1, :]
    if (pred_bb[:, 2:4] == 0.0).any():
        for i in range(1, pred_bb.shape[0]):
            if (pred_bb[i, 2:4] == 0.0).any() and not np.isnan(anno_bb[i, :]).any():
                pred_bb[i, :] = pred_bb[i-1, :]
    
    # 第59-72行: 长度不匹配处理
    if pred_bb.shape[0] != anno_bb.shape[0]:
        if pred_bb.shape[0] > anno_bb.shape[0]:
            pred_bb = pred_bb[:anno_bb.shape[0], :]
        else:
            pad = np.zeros((anno_bb.shape[0] - pred_bb.shape[0], 4), dtype=np.float64)
            pred_bb = np.vstack([pred_bb, pad])
    
    # 第74行: 第一帧使用GT
    # pred_bb[0, :] = anno_bb[0, :]
    pred_bb[0, :] = anno_bb[0, :]
    
    # 第79-80行: valid计算
    # valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2)
    valid = (anno_bb[:, 2:4] > 0.0).sum(axis=1) == 2
    
    # 第82-84行: 计算误差
    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)
    
    # 第86-92行: 处理无效帧
    # if dataset in ['uav']:
    #     err_center[~valid] = -1.0
    # else:
    #     err_center[~valid] = float("Inf")
    # err_center_normalized[~valid] = -1.0
    # err_overlap[~valid] = -1.0
    if dataset in ['uav']:
        err_center[~valid] = -1.0
    else:
        err_center[~valid] = np.inf
    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0
    
    return err_overlap, err_center, err_center_normalized, valid


def evaluate_sequence(pred_bb, anno_bb, dataset,
                      threshold_set_overlap, threshold_set_center, threshold_set_center_norm,
                      exclude_invalid_frames=False):
    """
    严格按照ORTrack的extract_results中的序列评测逻辑实现 (第155-171行)
    """
    err_overlap, err_center, err_center_normalized, valid_frame = calc_seq_err_robust(
        pred_bb, anno_bb, dataset)
    
    # 第161-164行: seq_length计算
    if exclude_invalid_frames:
        seq_length = valid_frame.sum()
    else:
        seq_length = anno_bb.shape[0]
    
    if seq_length <= 0:
        seq_length = 1  # 防止除零
    
    # 第169行: overlap用严格大于 (>)
    # ave_success_rate_plot_overlap[seq_id, trk_id, :] = (err_overlap.view(-1, 1) > threshold_set_overlap.view(1, -1)).sum(0).float() / seq_length
    success_rate_overlap = (err_overlap.reshape(-1, 1) > threshold_set_overlap.reshape(1, -1)).sum(axis=0).astype(np.float32) / seq_length
    
    # 第170行: center用小于等于 (<=)
    # ave_success_rate_plot_center[seq_id, trk_id, :] = (err_center.view(-1, 1) <= threshold_set_center.view(1, -1)).sum(0).float() / seq_length
    success_rate_center = (err_center.reshape(-1, 1) <= threshold_set_center.reshape(1, -1)).sum(axis=0).astype(np.float32) / seq_length
    
    # 第171行
    # ave_success_rate_plot_center_norm[seq_id, trk_id, :] = (err_center_normalized.view(-1, 1) <= threshold_set_center_norm.view(1, -1)).sum(0).float() / seq_length
    success_rate_center_norm = (err_center_normalized.reshape(-1, 1) <= threshold_set_center_norm.reshape(1, -1)).sum(axis=0).astype(np.float32) / seq_length
    
    return {
        'success_rate_overlap': success_rate_overlap,
        'success_rate_center': success_rate_center,
        'success_rate_center_norm': success_rate_center_norm,
        'err_overlap': err_overlap,
        'err_center': err_center,
        'err_center_normalized': err_center_normalized,
        'valid_frame': valid_frame
    }


def evaluate_tracker_on_dataset(
    tracker_results: Dict[str, np.ndarray],
    ground_truths: Dict[str, np.ndarray],
    method: str = 'otb',
    attributes: Dict[str, np.ndarray] = None,
    attribute_names: List[str] = None,
    dataset_name: str = ''
) -> Dict:
    """
    评测单个跟踪器在数据集上的表现
    严格按照ORTrack的extract_results和plot_results实现
    """
    
    # 第113-115行: 阈值设置
    # threshold_set_overlap = torch.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64)
    # threshold_set_center = torch.arange(0, 51, dtype=torch.float64)
    # threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0
    plot_bin_gap = 0.05
    threshold_set_overlap = np.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=np.float64)
    threshold_set_center = np.arange(0, 51, dtype=np.float64)
    threshold_set_center_norm = np.arange(0, 51, dtype=np.float64) / 100.0
    
    seq_names = list(ground_truths.keys())
    num_seqs = len(seq_names)
    
    # 第118-123行: 初始化数组
    ave_success_rate_plot_overlap = np.zeros((num_seqs, len(threshold_set_overlap)), dtype=np.float32)
    ave_success_rate_plot_center = np.zeros((num_seqs, len(threshold_set_center)), dtype=np.float32)
    ave_success_rate_plot_center_norm = np.zeros((num_seqs, len(threshold_set_center_norm)), dtype=np.float32)
    
    # 第125行: valid_sequence
    valid_sequence = np.ones(num_seqs, dtype=np.uint8)
    
    # 评测每个序列
    for seq_id, seq_name in enumerate(seq_names):
        if seq_name not in tracker_results:
            valid_sequence[seq_id] = 0
            continue
        
        pred_bb = tracker_results[seq_name].astype(np.float64)
        anno_bb = ground_truths[seq_name].astype(np.float64)
        
        # 使用dataset_name作为dataset参数
        result = evaluate_sequence(
            pred_bb, anno_bb, dataset_name,
            threshold_set_overlap, threshold_set_center, threshold_set_center_norm
        )
        
        ave_success_rate_plot_overlap[seq_id, :] = result['success_rate_overlap']
        ave_success_rate_plot_center[seq_id, :] = result['success_rate_center']
        ave_success_rate_plot_center_norm[seq_id, :] = result['success_rate_center_norm']
    
    # 计算总体指标 - 严格按照plot_results.py
    valid_mask = valid_sequence.astype(bool)
    
    # get_auc_curve (plot_results.py 第206-211行):
    # def get_auc_curve(ave_success_rate_plot_overlap, valid_sequence):
    #     ave_success_rate_plot_overlap = ave_success_rate_plot_overlap[valid_sequence, :, :]
    #     auc_curve = ave_success_rate_plot_overlap.mean(0) * 100.0
    #     auc = auc_curve.mean(-1)
    #     return auc_curve, auc
    if valid_mask.sum() > 0:
        auc_curve = ave_success_rate_plot_overlap[valid_mask, :].mean(axis=0) * 100.0
    else:
        auc_curve = np.zeros(len(threshold_set_overlap), dtype=np.float32)
    auc = float(auc_curve.mean())
    
    # get_prec_curve (plot_results.py 第214-219行):
    # def get_prec_curve(ave_success_rate_plot_center, valid_sequence):
    #     ave_success_rate_plot_center = ave_success_rate_plot_center[valid_sequence, :, :]
    #     prec_curve = ave_success_rate_plot_center.mean(0) * 100.0
    #     prec_score = prec_curve[:, 20]
    #     return prec_curve, prec_score
    if valid_mask.sum() > 0:
        prec_curve = ave_success_rate_plot_center[valid_mask, :].mean(axis=0) * 100.0
        norm_prec_curve = ave_success_rate_plot_center_norm[valid_mask, :].mean(axis=0) * 100.0
    else:
        prec_curve = np.zeros(len(threshold_set_center), dtype=np.float32)
        norm_prec_curve = np.zeros(len(threshold_set_center_norm), dtype=np.float32)
    
    # prec_score = prec_curve[:, 20] - 索引20对应threshold=20
    precision_20 = float(prec_curve[20]) if len(prec_curve) > 20 else 0.0
    norm_precision_20 = float(norm_prec_curve[20]) if len(norm_prec_curve) > 20 else 0.0
    
    # OP50, OP75
    # scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
    # scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]
    # threshold_set_overlap[10] = 0.50, threshold_set_overlap[15] = 0.75
    success_50 = float(auc_curve[10]) if len(auc_curve) > 10 else 0.0
    success_75 = float(auc_curve[15]) if len(auc_curve) > 15 else 0.0
    
    results = {
        'method': method,
        'threshold_overlap': threshold_set_overlap.tolist(),
        'threshold_center': threshold_set_center.tolist(),
        'threshold_center_norm': threshold_set_center_norm.tolist(),
        'all_success_overlap': ave_success_rate_plot_overlap.tolist(),
        'all_success_center': ave_success_rate_plot_center.tolist(),
        'all_success_center_norm': ave_success_rate_plot_center_norm.tolist(),
        'avg_success_overlap': (auc_curve / 100.0).tolist(),  # 保持0-1范围
        'avg_success_center': (prec_curve / 100.0).tolist(),
        'avg_success_center_norm': (norm_prec_curve / 100.0).tolist(),
        'auc': auc,
        'precision_20': precision_20,
        'norm_precision_20': norm_precision_20,
        'success_50': success_50,
        'success_75': success_75,
    }
    
    # 计算属性结果
    if attributes is not None and attribute_names is not None and len(attribute_names) > 0:
        att_results = {}
        
        # 尝试匹配序列名
        attr_seq_names = set(attributes.keys())
        gt_seq_names = set(seq_names)
        matched_seqs = attr_seq_names & gt_seq_names
        
        if len(matched_seqs) == 0:
            attr_seq_lower = {k.lower(): k for k in attr_seq_names}
            new_attributes = {}
            for gt_name in seq_names:
                gt_lower = gt_name.lower()
                if gt_lower in attr_seq_lower:
                    orig_attr_name = attr_seq_lower[gt_lower]
                    new_attributes[gt_name] = attributes[orig_attr_name]
            if new_attributes:
                attributes = new_attributes
        
        for att_idx, att_name in enumerate(attribute_names):
            att_seq_indices = []
            for idx, seq_name in enumerate(seq_names):
                if seq_name in attributes:
                    att_vec = attributes[seq_name]
                    if att_idx < len(att_vec) and att_vec[att_idx] > 0:
                        att_seq_indices.append(idx)
            
            if len(att_seq_indices) == 0:
                continue
            
            att_seq_indices = np.array(att_seq_indices)
            att_valid = valid_sequence[att_seq_indices].astype(bool)
            
            if att_valid.sum() == 0:
                continue
            
            # 计算该属性的指标
            att_auc_curve = ave_success_rate_plot_overlap[att_seq_indices][att_valid].mean(axis=0) * 100.0
            att_prec_curve = ave_success_rate_plot_center[att_seq_indices][att_valid].mean(axis=0) * 100.0
            att_norm_prec_curve = ave_success_rate_plot_center_norm[att_seq_indices][att_valid].mean(axis=0) * 100.0
            
            att_results[att_name] = {
                'num_seqs': int(att_valid.sum()),
                'avg_success_overlap': (att_auc_curve / 100.0).tolist(),
                'avg_success_center': (att_prec_curve / 100.0).tolist(),
                'avg_success_center_norm': (att_norm_prec_curve / 100.0).tolist(),
                'auc': float(att_auc_curve.mean()),
                'precision_20': float(att_prec_curve[20]) if len(att_prec_curve) > 20 else 0.0,
                'norm_precision_20': float(att_norm_prec_curve[20]) if len(att_norm_prec_curve) > 20 else 0.0
            }
        
        results['attribute_results'] = att_results
    
    return results


def evaluate_multiple_trackers(
    all_tracker_results: Dict[str, Dict[str, np.ndarray]],
    ground_truths: Dict[str, np.ndarray],
    method: str = 'otb',
    attributes: Dict[str, np.ndarray] = None,
    attribute_names: List[str] = None,
    dataset_name: str = ''
) -> Dict[str, Dict]:
    """评测多个跟踪器"""
    all_results = {}
    for tracker_name, tracker_results in all_tracker_results.items():
        all_results[tracker_name] = evaluate_tracker_on_dataset(
            tracker_results, ground_truths, method, attributes, attribute_names, dataset_name
        )
    return all_results


# =============================================================================
# OTB方法 (保持兼容)
# =============================================================================

def calc_rect_int_otb(pred_bb: np.ndarray, gt_bb: np.ndarray) -> np.ndarray:
    """OTB方式计算IoU"""
    leftA = pred_bb[:, 0]
    bottomA = pred_bb[:, 1]
    rightA = leftA + pred_bb[:, 2] - 1
    topA = bottomA + pred_bb[:, 3] - 1
    
    leftB = gt_bb[:, 0]
    bottomB = gt_bb[:, 1]
    rightB = leftB + gt_bb[:, 2] - 1
    topB = bottomB + gt_bb[:, 3] - 1
    
    intersect_w = np.maximum(0, np.minimum(rightA, rightB) - np.maximum(leftA, leftB) + 1)
    intersect_h = np.maximum(0, np.minimum(topA, topB) - np.maximum(bottomA, bottomB) + 1)
    intersection = intersect_w * intersect_h
    
    areaA = pred_bb[:, 2] * pred_bb[:, 3]
    areaB = gt_bb[:, 2] * gt_bb[:, 3]
    union = areaA + areaB - intersection
    
    overlap = np.where(union > 0, intersection / union, 0.0)
    return overlap


def calc_center_error_otb(pred_bb: np.ndarray, gt_bb: np.ndarray) -> np.ndarray:
    """OTB方式计算中心误差"""
    pred_cx = pred_bb[:, 0] + (pred_bb[:, 2] - 1) / 2.0
    pred_cy = pred_bb[:, 1] + (pred_bb[:, 3] - 1) / 2.0
    
    gt_cx = gt_bb[:, 0] + (gt_bb[:, 2] - 1) / 2.0
    gt_cy = gt_bb[:, 1] + (gt_bb[:, 3] - 1) / 2.0
    
    err_center = np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)
    return err_center


def compute_sequence_iou(pred_bb: np.ndarray, gt_bb: np.ndarray) -> np.ndarray:
    """计算单个序列每帧的IoU值"""
    pred_bb = pred_bb.copy().astype(np.float64)
    gt_bb = gt_bb.copy().astype(np.float64)
    seq_length = len(gt_bb)
    
    if len(pred_bb) < seq_length:
        pad = np.zeros((seq_length - len(pred_bb), 4))
        pred_bb = np.vstack([pred_bb, pad])
    elif len(pred_bb) > seq_length:
        pred_bb = pred_bb[:seq_length]
    
    for i in range(1, seq_length):
        if (pred_bb[i, 2:4] == 0.0).any() and not np.isnan(gt_bb[i]).any():
            pred_bb[i] = pred_bb[i-1]
    
    pred_bb[0] = gt_bb[0]
    
    iou_per_frame = calc_rect_int_otb(pred_bb, gt_bb)
    
    valid_idx = (gt_bb[:, 2:4] > 0).sum(axis=1) == 2
    iou_per_frame[~valid_idx] = -1.0
    
    return iou_per_frame


def compute_sequence_center_error(pred_bb: np.ndarray, gt_bb: np.ndarray) -> np.ndarray:
    """计算单个序列每帧的中心误差"""
    pred_bb = pred_bb.copy().astype(np.float64)
    gt_bb = gt_bb.copy().astype(np.float64)
    seq_length = len(gt_bb)
    
    if len(pred_bb) < seq_length:
        pad = np.zeros((seq_length - len(pred_bb), 4))
        pred_bb = np.vstack([pred_bb, pad])
    elif len(pred_bb) > seq_length:
        pred_bb = pred_bb[:seq_length]
    
    for i in range(1, seq_length):
        if (pred_bb[i, 2:4] == 0.0).any() and not np.isnan(gt_bb[i]).any():
            pred_bb[i] = pred_bb[i-1]
    
    pred_bb[0] = gt_bb[0]
    
    center_error = calc_center_error_otb(pred_bb, gt_bb)
    
    valid_idx = (gt_bb[:, 2:4] > 0).sum(axis=1) == 2
    center_error[~valid_idx] = -1.0
    
    return center_error
