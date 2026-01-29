"""
Tracker Benchmark 可视化模块
生成精度图、成功率图、属性图和雷达图
保持OTB原始风格
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import io
import base64

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# OTB风格的颜色和线型
PLOT_STYLES = [
    {'color': (1, 0, 0), 'linestyle': '-'},
    {'color': (0, 1, 0), 'linestyle': '-'},
    {'color': (0, 0, 1), 'linestyle': '-'},
    {'color': (0, 0, 0), 'linestyle': '-'},
    {'color': (1, 0, 1), 'linestyle': '-'},
    {'color': (0, 1, 1), 'linestyle': '-'},
    {'color': (0.5, 0.5, 0.5), 'linestyle': '-'},
    {'color': (136/255, 0, 21/255), 'linestyle': '-'},
    {'color': (255/255, 127/255, 39/255), 'linestyle': '-'},
    {'color': (0, 162/255, 232/255), 'linestyle': '-'},
    {'color': (163/255, 73/255, 164/255), 'linestyle': '-'},
    {'color': (1, 0, 0), 'linestyle': '--'},
    {'color': (0, 1, 0), 'linestyle': '--'},
    {'color': (0, 0, 1), 'linestyle': '--'},
    {'color': (0, 0, 0), 'linestyle': '--'},
    {'color': (1, 0, 1), 'linestyle': '--'},
    {'color': (0, 1, 1), 'linestyle': '--'},
    {'color': (0.5, 0.5, 0.5), 'linestyle': '--'},
    {'color': (136/255, 0, 21/255), 'linestyle': '--'},
    {'color': (255/255, 127/255, 39/255), 'linestyle': '--'},
    {'color': (0, 162/255, 232/255), 'linestyle': '--'},
    {'color': (163/255, 73/255, 164/255), 'linestyle': '--'},
    {'color': (1, 0, 0), 'linestyle': '-.'},
    {'color': (0, 1, 0), 'linestyle': '-.'},
    {'color': (0, 0, 1), 'linestyle': '-.'},
    {'color': (0, 0, 0), 'linestyle': '-.'},
    {'color': (1, 0, 1), 'linestyle': '-.'},
    {'color': (0, 1, 1), 'linestyle': '-.'},
    {'color': (0.5, 0.5, 0.5), 'linestyle': '-.'},
    {'color': (136/255, 0, 21/255), 'linestyle': '-.'},
    {'color': (255/255, 127/255, 39/255), 'linestyle': '-.'},
    {'color': (0, 162/255, 232/255), 'linestyle': '-.'},
]


def fig_to_base64(fig) -> str:
    """将matplotlib图转换为base64字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64


def plot_success_curve(tracker_results: Dict[str, Dict],
                       title: str = '成功率曲线',
                       rank_num: int = 10,
                       legend_fontsize: int = 10,
                       tracker_colors: Dict[str, str] = None) -> str:
    """绘制成功率曲线"""
    
    # 获取排名
    scores = []
    for name, result in tracker_results.items():
        score = result.get('auc', 0)
        scores.append((name, score, result))
    scores.sort(key=lambda x: x[1], reverse=True)
    
    if rank_num > 0:
        scores = scores[:min(rank_num, len(scores))]
    
    if len(scores) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    for i, (name, score, result) in enumerate(scores):
        y_data = np.array(result.get('avg_success_overlap', []))
        if len(y_data) == 0:
            continue
        
        x_data = np.array(result.get('threshold_overlap', np.arange(0, 1.05, 0.05)))
        
        # 获取颜色
        if tracker_colors and name in tracker_colors:
            color = tracker_colors[name]
            linestyle = '-'
        else:
            style = PLOT_STYLES[i % len(PLOT_STYLES)]
            color = style['color']
            linestyle = style['linestyle']
        
        if result.get('method') == 'ortrack':
            label = f'{name} [{score:.1f}]'
        else:
            label = f'{name} [{score:.3f}]'
        
        ax.plot(x_data, y_data, color=color, linestyle=linestyle,
                linewidth=2, label=label)
    
    ax.set_xlabel('Overlap threshold', fontsize=14)
    ax.set_ylabel('Success rate', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.legend(loc='lower left', fontsize=legend_fontsize, framealpha=0.9)
    plt.tight_layout()
    
    return fig_to_base64(fig)


def plot_precision_curve(tracker_results: Dict[str, Dict],
                         title: str = '精度曲线',
                         rank_num: int = 10,
                         legend_fontsize: int = 10,
                         tracker_colors: Dict[str, str] = None) -> str:
    """绘制精度曲线"""
    
    scores = []
    for name, result in tracker_results.items():
        score = result.get('precision_20', 0)
        scores.append((name, score, result))
    scores.sort(key=lambda x: x[1], reverse=True)
    
    if rank_num > 0:
        scores = scores[:min(rank_num, len(scores))]
    
    if len(scores) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    for i, (name, score, result) in enumerate(scores):
        # OTB用avg_success_error, ORTrack用avg_success_center
        if 'avg_success_error' in result:
            y_data = np.array(result['avg_success_error'])
            x_data = np.array(result.get('threshold_error', np.arange(0, 51)))
        elif 'avg_success_center' in result:
            y_data = np.array(result['avg_success_center'])
            x_data = np.array(result.get('threshold_center', np.arange(0, 51)))
        else:
            continue
        
        if len(y_data) == 0:
            continue
        
        # 获取颜色
        if tracker_colors and name in tracker_colors:
            color = tracker_colors[name]
            linestyle = '-'
        else:
            style = PLOT_STYLES[i % len(PLOT_STYLES)]
            color = style['color']
            linestyle = style['linestyle']
        
        if result.get('method') == 'ortrack':
            label = f'{name} [{score:.1f}]'
        else:
            label = f'{name} [{score:.3f}]'
        
        ax.plot(x_data, y_data, color=color, linestyle=linestyle,
                linewidth=2, label=label)
    
    ax.set_xlabel('Location error threshold', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.legend(loc='lower right', fontsize=legend_fontsize, framealpha=0.9)
    plt.tight_layout()
    
    return fig_to_base64(fig)


def plot_norm_precision_curve(tracker_results: Dict[str, Dict],
                              title: str = '归一化精度曲线',
                              rank_num: int = 10,
                              legend_fontsize: int = 10,
                              tracker_colors: Dict[str, str] = None) -> str:
    """绘制归一化精度曲线（仅ORTrack）"""
    
    scores = []
    for name, result in tracker_results.items():
        score = result.get('norm_precision_20', 0)
        scores.append((name, score, result))
    scores.sort(key=lambda x: x[1], reverse=True)
    
    if rank_num > 0:
        scores = scores[:min(rank_num, len(scores))]
    
    if len(scores) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    for i, (name, score, result) in enumerate(scores):
        y_data = np.array(result.get('avg_success_center_norm', []))
        if len(y_data) == 0:
            continue
        
        x_data = np.array(result.get('threshold_center_norm', np.arange(0, 51) / 100.0))
        
        # 获取颜色
        if tracker_colors and name in tracker_colors:
            color = tracker_colors[name]
            linestyle = '-'
        else:
            style = PLOT_STYLES[i % len(PLOT_STYLES)]
            color = style['color']
            linestyle = style['linestyle']
        
        label = f'{name} [{score:.1f}]'
        
        ax.plot(x_data, y_data, color=color, linestyle=linestyle,
                linewidth=2, label=label)
    
    ax.set_xlabel('Normalized location error threshold', fontsize=14)
    ax.set_ylabel('Normalized Precision', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.legend(loc='lower right', fontsize=legend_fontsize, framealpha=0.9)
    plt.tight_layout()
    
    return fig_to_base64(fig)


def plot_attribute_success(tracker_results: Dict[str, Dict],
                           attribute_name: str,
                           rank_num: int = 10,
                           legend_fontsize: int = 10,
                           tracker_colors: Dict[str, str] = None) -> str:
    """绘制特定属性的成功率曲线"""
    
    # 收集有该属性结果的tracker
    scores = []
    for name, result in tracker_results.items():
        att_results = result.get('attribute_results', {})
        if attribute_name not in att_results:
            continue
        att_data = att_results[attribute_name]
        score = att_data.get('auc', 0)
        scores.append((name, score, result, att_data))
    
    if len(scores) == 0:
        return None
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    if rank_num > 0:
        scores = scores[:min(rank_num, len(scores))]
    
    # 获取序列数量
    num_seqs = scores[0][3].get('num_seqs', 0) if scores else 0
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    for i, (name, score, result, att_data) in enumerate(scores):
        y_data = np.array(att_data.get('avg_success_overlap', []))
        if len(y_data) == 0:
            continue
        
        x_data = np.array(result.get('threshold_overlap', np.arange(0, 1.05, 0.05)))
        
        # 获取颜色
        if tracker_colors and name in tracker_colors:
            color = tracker_colors[name]
            linestyle = '-'
        else:
            style = PLOT_STYLES[i % len(PLOT_STYLES)]
            color = style['color']
            linestyle = style['linestyle']
        
        if result.get('method') == 'ortrack':
            label = f'{name} [{score:.1f}]'
        else:
            label = f'{name} [{score:.3f}]'
        
        ax.plot(x_data, y_data, color=color, linestyle=linestyle,
                linewidth=2, label=label)
    
    ax.set_xlabel('Overlap threshold', fontsize=14)
    ax.set_ylabel('Success rate', fontsize=14)
    ax.set_title(f'Success plots - {attribute_name} ({num_seqs})', fontsize=16)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.legend(loc='lower left', fontsize=legend_fontsize, framealpha=0.9)
    plt.tight_layout()
    
    return fig_to_base64(fig)


def plot_attribute_precision(tracker_results: Dict[str, Dict],
                             attribute_name: str,
                             rank_num: int = 10,
                             legend_fontsize: int = 10,
                             tracker_colors: Dict[str, str] = None) -> str:
    """绘制特定属性的精度曲线"""
    
    scores = []
    for name, result in tracker_results.items():
        att_results = result.get('attribute_results', {})
        if attribute_name not in att_results:
            continue
        att_data = att_results[attribute_name]
        score = att_data.get('precision_20', 0)
        scores.append((name, score, result, att_data))
    
    if len(scores) == 0:
        return None
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    if rank_num > 0:
        scores = scores[:min(rank_num, len(scores))]
    
    num_seqs = scores[0][3].get('num_seqs', 0) if scores else 0
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    for i, (name, score, result, att_data) in enumerate(scores):
        # OTB用avg_success_error, ORTrack用avg_success_center
        if 'avg_success_error' in att_data:
            y_data = np.array(att_data['avg_success_error'])
            x_data = np.array(result.get('threshold_error', np.arange(0, 51)))
        elif 'avg_success_center' in att_data:
            y_data = np.array(att_data['avg_success_center'])
            x_data = np.array(result.get('threshold_center', np.arange(0, 51)))
        else:
            continue
        
        if len(y_data) == 0:
            continue
        
        # 获取颜色
        if tracker_colors and name in tracker_colors:
            color = tracker_colors[name]
            linestyle = '-'
        else:
            style = PLOT_STYLES[i % len(PLOT_STYLES)]
            color = style['color']
            linestyle = style['linestyle']
        
        if result.get('method') == 'ortrack':
            label = f'{name} [{score:.1f}]'
        else:
            label = f'{name} [{score:.3f}]'
        
        ax.plot(x_data, y_data, color=color, linestyle=linestyle,
                linewidth=2, label=label)
    
    ax.set_xlabel('Location error threshold', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title(f'Precision plots - {attribute_name} ({num_seqs})', fontsize=16)
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.legend(loc='lower right', fontsize=legend_fontsize, framealpha=0.9)
    plt.tight_layout()
    
    return fig_to_base64(fig)


def plot_radar_chart(tracker_results: Dict[str, Dict],
                     attribute_names: List[str],
                     metric: str = 'auc',
                     title: str = 'Radar Chart',
                     tracker_colors: Dict[str, str] = None,
                     legend_fontsize: int = 10) -> str:
    """绘制雷达图"""
    
    if not attribute_names or len(attribute_names) < 3:
        return None
    
    # 收集数据
    tracker_data = []
    for name, result in tracker_results.items():
        att_results = result.get('attribute_results', {})
        if not att_results:
            continue
        
        values = []
        valid = True
        for attr in attribute_names:
            if attr in att_results:
                val = att_results[attr].get(metric, 0)
                values.append(val)
            else:
                valid = False
                break
        
        if valid and len(values) == len(attribute_names):
            tracker_data.append((name, values))
    
    if len(tracker_data) == 0:
        return None
    
    num_attrs = len(attribute_names)
    angles = np.linspace(0, 2 * np.pi, num_attrs, endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    
    for i, (name, values) in enumerate(tracker_data):
        values_plot = values + values[:1]  # 闭合
        
        if tracker_colors and name in tracker_colors:
            color = tracker_colors[name]
        else:
            style = PLOT_STYLES[i % len(PLOT_STYLES)]
            color = style['color']
        
        ax.plot(angles, values_plot, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values_plot, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attribute_names, fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=legend_fontsize)
    ax.set_title(title, fontsize=16, pad=20)
    
    plt.tight_layout()
    
    return fig_to_base64(fig)


def plot_sequence_iou_curve(iou_data: Dict[str, np.ndarray],
                            seq_name: str,
                            title: str = None,
                            tracker_colors: Dict[str, str] = None,
                            legend_fontsize: int = 10) -> str:
    """
    绘制单个序列的IoU曲线对比图
    
    参数:
        iou_data: {tracker_name: iou_per_frame} 每个跟踪器在该序列上每帧的IoU
        seq_name: 序列名称
        tracker_colors: {tracker_name: color} 自定义颜色
        legend_fontsize: 图例字体大小
    
    返回:
        base64编码的图片
    """
    if not iou_data:
        return None
    
    if title is None:
        title = f'IoU Curve - {seq_name}'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 按平均IoU排序
    sorted_trackers = sorted(iou_data.items(), 
                             key=lambda x: np.mean(x[1][x[1] >= 0]) if np.any(x[1] >= 0) else 0,
                             reverse=True)
    
    for i, (tracker_name, iou_values) in enumerate(sorted_trackers):
        # 过滤有效帧
        valid_mask = iou_values >= 0
        if not np.any(valid_mask):
            continue
        
        frames = np.arange(len(iou_values))
        
        # 获取颜色
        if tracker_colors and tracker_name in tracker_colors:
            color = tracker_colors[tracker_name]
        else:
            style = PLOT_STYLES[i % len(PLOT_STYLES)]
            color = style['color']
        
        # 计算平均IoU（仅有效帧）
        avg_iou = np.mean(iou_values[valid_mask])
        
        # 绘制曲线
        ax.plot(frames, iou_values, color=color, linewidth=1.5, 
                label=f'{tracker_name} [avg: {avg_iou:.3f}]', alpha=0.8)
    
    ax.set_xlabel('Frame', fontsize=14)
    ax.set_ylabel('IoU (Overlap)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim([0, len(list(iou_data.values())[0]) - 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.legend(loc='lower left', fontsize=legend_fontsize, framealpha=0.9)
    
    plt.tight_layout()
    
    return fig_to_base64(fig)


def plot_sequence_center_error_curve(error_data: Dict[str, np.ndarray],
                                      seq_name: str,
                                      title: str = None,
                                      tracker_colors: Dict[str, str] = None,
                                      legend_fontsize: int = 10) -> str:
    """
    绘制单个序列的中心误差曲线对比图
    
    参数:
        error_data: {tracker_name: center_error_per_frame}
        seq_name: 序列名称
        tracker_colors: 自定义颜色
        legend_fontsize: 图例字体大小
    
    返回:
        base64编码的图片
    """
    if not error_data:
        return None
    
    if title is None:
        title = f'Center Error Curve - {seq_name}'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 按平均误差排序（误差越小越好）
    sorted_trackers = sorted(error_data.items(), 
                             key=lambda x: np.mean(x[1][x[1] >= 0]) if np.any(x[1] >= 0) else float('inf'))
    
    max_error = 0
    for i, (tracker_name, error_values) in enumerate(sorted_trackers):
        valid_mask = error_values >= 0
        if not np.any(valid_mask):
            continue
        
        frames = np.arange(len(error_values))
        
        if tracker_colors and tracker_name in tracker_colors:
            color = tracker_colors[tracker_name]
        else:
            style = PLOT_STYLES[i % len(PLOT_STYLES)]
            color = style['color']
        
        avg_error = np.mean(error_values[valid_mask])
        max_error = max(max_error, np.max(error_values[valid_mask]))
        
        ax.plot(frames, error_values, color=color, linewidth=1.5, 
                label=f'{tracker_name} [avg: {avg_error:.1f}px]', alpha=0.8)
    
    ax.set_xlabel('Frame', fontsize=14)
    ax.set_ylabel('Center Error (pixels)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim([0, len(list(error_data.values())[0]) - 1])
    ax.set_ylim([0, min(max_error * 1.1, 100)])  # 限制最大显示100像素
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.legend(loc='upper right', fontsize=legend_fontsize, framealpha=0.9)
    
    plt.tight_layout()
    
    return fig_to_base64(fig)


def generate_all_plots(tracker_results: Dict[str, Dict],
                       attribute_names: List[str] = None,
                       output_dir: str = None,
                       dataset_name: str = 'Dataset',
                       method: str = 'otb',
                       legend_fontsize: int = 10,
                       rank_num: int = 10) -> Dict[str, str]:
    """生成所有图表"""
    import os
    
    plots = {}
    
    # 总体成功率图
    img = plot_success_curve(tracker_results, f'Success plots of OPE - {dataset_name}',
                             rank_num, legend_fontsize)
    if img:
        if output_dir:
            path = os.path.join(output_dir, 'success_overall.png')
            with open(path, 'wb') as f:
                f.write(base64.b64decode(img))
            plots['success_overall'] = path
        else:
            plots['success_overall'] = img
    
    # 总体精度图
    img = plot_precision_curve(tracker_results, f'Precision plots of OPE - {dataset_name}',
                               rank_num, legend_fontsize)
    if img:
        if output_dir:
            path = os.path.join(output_dir, 'precision_overall.png')
            with open(path, 'wb') as f:
                f.write(base64.b64decode(img))
            plots['precision_overall'] = path
        else:
            plots['precision_overall'] = img
    
    # 归一化精度图（仅ORTrack）
    if method == 'ortrack':
        img = plot_norm_precision_curve(tracker_results, f'Normalized Precision - {dataset_name}',
                                        rank_num, legend_fontsize)
        if img:
            if output_dir:
                path = os.path.join(output_dir, 'norm_precision_overall.png')
                with open(path, 'wb') as f:
                    f.write(base64.b64decode(img))
                plots['norm_precision_overall'] = path
            else:
                plots['norm_precision_overall'] = img
    
    # 属性图
    if attribute_names:
        for attr in attribute_names:
            # 检查是否有该属性的数据
            has_attr = False
            for result in tracker_results.values():
                if 'attribute_results' in result and attr in result['attribute_results']:
                    has_attr = True
                    break
            
            if not has_attr:
                continue
            
            attr_safe = attr.replace(' ', '_').replace('/', '_')
            
            img = plot_attribute_success(tracker_results, attr, rank_num, legend_fontsize)
            if img:
                if output_dir:
                    path = os.path.join(output_dir, f'success_{attr_safe}.png')
                    with open(path, 'wb') as f:
                        f.write(base64.b64decode(img))
                    plots[f'success_{attr_safe}'] = path
                else:
                    plots[f'success_{attr_safe}'] = img
            
            img = plot_attribute_precision(tracker_results, attr, rank_num, legend_fontsize)
            if img:
                if output_dir:
                    path = os.path.join(output_dir, f'precision_{attr_safe}.png')
                    with open(path, 'wb') as f:
                        f.write(base64.b64decode(img))
                    plots[f'precision_{attr_safe}'] = path
                else:
                    plots[f'precision_{attr_safe}'] = img
    
    return plots
