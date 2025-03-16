import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import argparse  # 使用 argparse 代替 HfArgumentParser


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--human_fname",
        type=str,
        default="outputs_human",
        help="File name of human code detection results",
    )
    parser.add_argument(
        "--machine_fname",
        type=str,
        default="outputs",
        help="File name of machine code detection results",
    )
    parser.add_argument(
        "--data_path",
        type=str,
         default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_eval_cp/gen_table_w_metrics.jsonl",
        help="Path to the data file containing the z-scores"
    )
    return parser.parse_args()


def load_z_scores(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    # 提取相关字段
    human_z_scores = [entry['no_wm_output_z_score'] for entry in data]
    machine_z_scores = [entry['w_wm_output_attacked_z_score'] for entry in data]
    text_lengths = [entry['w_wm_output_length'] for entry in data]  # 使用 'w_wm_output_length' 作为文本长度
    
    return human_z_scores, machine_z_scores, text_lengths

def clean_z_scores(human_z, machine_z):
    # Filter out any NaN values from both human_z and machine_z
    valid_human_z = [z for z in human_z if not np.isnan(z)]
    valid_machine_z = [z for z in machine_z if not np.isnan(z)]
    # Ensure both lists are of the same length
    min_len = min(len(valid_human_z), len(valid_machine_z))
    return valid_human_z[:min_len], valid_machine_z[:min_len]

def get_roc_auc(human_z, machine_z):
    assert len(human_z) == len(machine_z)
    # 合并人类和机器的 z-score
    human_z, machine_z = clean_z_scores(human_z, machine_z)
    all_scores = np.concatenate([np.array(human_z), np.array(machine_z)])
    
    # 生成标签（0 为人类，1 为机器）
    baseline_labels = np.zeros_like(human_z)
    watermark_labels = np.ones_like(machine_z)
    all_labels = np.concatenate([baseline_labels, watermark_labels])

    # 计算 ROC 曲线和 AUC
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc, fpr, tpr, thresholds


def plot_auc_vs_length(human_z, machine_z, text_lengths):
    # 定义文本长度区间（可以根据实际情况调整）
    length_bins = np.arange(0, max(text_lengths) + 100, 10)
    
    auc_values = []
    length_centers = []
    
    for i in range(len(length_bins) - 1):
        # 获取落入当前区间的文本长度
        bin_start = length_bins[i]
        bin_end = length_bins[i + 1]
        
        # 筛选出落入该区间的样本索引
        indices = [idx for idx, length in enumerate(text_lengths) if bin_start <= length < bin_end]
        
        if not indices:
            continue
        
        # 获取该区间的 z-scores
        human_z_bin = [human_z[idx] for idx in indices]
        machine_z_bin = [machine_z[idx] for idx in indices]
        
        # 计算该区间的 AUC-ROC
        roc_auc, _, _, _ = get_roc_auc(human_z_bin, machine_z_bin)
        
        auc_values.append(roc_auc)
        length_centers.append((bin_start + bin_end) / 2)  # 以区间的中点作为横坐标

    # 绘制 AUC vs. 文本长度的曲线
    plt.plot(length_centers, auc_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Text Length")
    plt.ylabel("AUC-ROC")
    plt.title("AUC-ROC vs Text Length")
    plt.grid(True)
    plt.savefig('/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/test_script/fig/AUCROC-Length.png')


def main():
    args = parse_args()

    # 加载 z-scores 和文本长度
    human_z, machine_z, text_lengths = load_z_scores(args.data_path)

    # 绘制 AUC-ROC 随文本长度变化的曲线图
    plot_auc_vs_length(human_z, machine_z, text_lengths)


if __name__ == "__main__":
    main()
