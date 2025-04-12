
import sys
sys.path.append("/home/shenhm/documents/lm-watermarking/watermark_reliability_release")
from watermark_processor import WatermarkDetector
import os
import json
from copy import deepcopy
from types import NoneType

from typing import Union
import numpy as np
import sklearn.metrics as metrics
import argparse  
import torch
from utils.submitit import str2bool  # better bool flag type for argparse
from functools import partial
from dataclasses import dataclass
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import softmax
import pdb
from tqdm import tqdm
import math

import json

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict

from argparse import ArgumentParser

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/windows_text/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6__0.25_LSH_H_6_c4/gen_table_dipper_O60_L60.jsonl",
        help="Path to the data file containing the z-scores"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/windows_text/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6__0.25_LSH_H_6_c4/gen_table_meta.json",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="ff-anchored_minhash_prf-6-True-15485863",
        help="The seeding procedure to use for the watermark.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default="0.25",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--n_hashes",
        type=int,
        default=5,
    )    
    parser.add_argument(
        "--n_features",
        type=int,
        default=32,
    )    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
    )
    # parser.add_argument(
    #     "--normalizers",
    #     type=Union[str, NoneType],
    #     default=None,
    #     help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    # )
    parser.add_argument(
        "--ignore_repeated_ngrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument('--bins', type=int, default=10, help="Number of length intervals to divide into")
    parser.add_argument('--max_tokens', type=int, default=150, help="Max token length to consider")
    parser.add_argument('--zscore_data_path', type=str, required=True, help="Path to save z-scores data")
    parser.add_argument('--plot_path', type=str, default='z_score_by_length.png')
    return parser.parse_args()

def check_missing_z_scores(data):
    missing_count = 0
    for idx, entry in enumerate(data):
        # 检查 'w_wm_output_z_score' 是否存在
        if 'w_wm_output_z_score' not in entry:
            missing_count += 1
            print(f"Missing 'w_wm_output_z_score' in entry {idx}: {entry}")
        
        # 检查其他可能缺少的 z_score 字段
        if 'w_wm_output_attacked_z_score' not in entry:
            missing_count += 1
            print(f"Missing 'w_wm_output_attacked_z_score' in entry {idx}: ")
        
        if 'no_wm_output_z_score' not in entry:
            missing_count += 1
            print(f"Missing 'no_wm_output_z_score' in entry {idx}")
    
    # 输出缺少 z_score 的条目数量
    if missing_count > 0:
        print(f"\nTotal {missing_count} entries are missing z_score fields.")
    else:
        print("\nAll entries have the required z_score fields.")


def main():
    args = parse_args()
    
    # 加载配置文件
    with open(args.config_path, 'r', encoding='utf-8') as infile:
        config_data = json.load(infile)
    
    tokenizer = AutoTokenizer.from_pretrained(config_data.get('model_name_or_path'), local_files_only=True)
    
    
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=config_data.get('gamma'),
        seeding_scheme=config_data.get('seeding_scheme'),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        tokenizer=tokenizer,
        z_threshold=args.detection_z_threshold,
        ignore_repeated_ngrams=args.ignore_repeated_ngrams,
    )
    
    # 加载数据
    with open(args.data_path, 'r', encoding='utf-8') as file:
        raw_data = [json.loads(line.strip()) for line in file]
    
    # 统计文本的token长度
    token_length_data = []
    for item in raw_data:
        for key in ["w_wm_output", "w_wm_output_attacked", "no_wm_output"]:
            input_text = item.get(key, "")
            if input_text:
                tokens = tokenizer.encode(input_text, truncation=False, add_special_tokens=False)
                token_length = len(tokens)
                token_length_data.append({
                    "item": item,
                    "key": key,
                    "text": input_text,
                    "length": min(token_length, args.max_tokens)  # 限制最大长度
                })

    # Step 1: 排序并将数据分成 N/d 个区间，保证各区间数据数量尽量相等
    token_length_data.sort(key=lambda x: x["length"])
    bin_size = len(token_length_data) // args.bins
    binned_data = []
    
    # 将数据分为多个区间
    for i in range(args.bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < args.bins - 1 else len(token_length_data)
        binned_data.append(token_length_data[start:end])

    # Step 2: 计算 z-score 并保存
    z_score_data = []
    for b_idx, bin_group in enumerate(binned_data):
        scores = []
        lengths = []
        max_len_in_bin = max(item["length"] for item in bin_group)
        
        for entry in bin_group:
            # 截取最大长度的 token
            truncated_tokens = tokenizer.encode(entry["text"], truncation=True, max_length=max_len_in_bin, add_special_tokens=False)
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
            try:
                score_dict = watermark_detector.detect(truncated_text, return_prediction=False, convert_to_float=True)
                z_score = score_dict.get("z_score", math.nan)
            except ValueError as e:
                z_score = math.nan if "Must have at least 1 token" in str(e) else math.nan
            
            scores.append(z_score)
            lengths.append(len(truncated_tokens))

        z_score_data.append({
            "bin_index": b_idx,
            "avg_length": np.mean(lengths) if lengths else 0,
            "z_scores": scores
        })
    
    # 保存 z-score 数据到本地
    zscore_data_path = '/home/shenhm/documents/lm-watermarking/experiments/bash/len_test.json'
    with open(zscore_data_path, 'w', encoding='utf-8') as f:
        json.dump(z_score_data, f, ensure_ascii=False, indent=4)
    print(f"Z-score data saved to: {zscore_data_path}")
    
    # 从本地读取 z-score 数据
    with open(args.zscore_data_path, 'r', encoding='utf-8') as f:
        loaded_z_score_data = json.load(f)
    
    # Step 3: 可视化
    avg_z_scores = [np.mean(bin_data['z_scores']) if bin_data['z_scores'] else 0 for bin_data in loaded_z_score_data]
    std_z_scores = [np.std(bin_data['z_scores']) if bin_data['z_scores'] else 0 for bin_data in loaded_z_score_data]
    avg_lengths = [bin_data['avg_length'] for bin_data in loaded_z_score_data]
    
    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.errorbar(avg_lengths, avg_z_scores, yerr=std_z_scores, fmt='o-', capsize=5)
    plt.title("Z-score vs Text Length")
    plt.xlabel("Average Token Length in Bin")
    plt.ylabel("Average Z-score ± Std")
    plt.grid(True)
    plt.savefig(args.plot_path)
    print(f"Z-score length analysis saved to {args.plot_path}")

if __name__ == '__main__':
    main()
