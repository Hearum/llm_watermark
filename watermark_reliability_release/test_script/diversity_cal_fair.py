import sys
sys.path.append("/home/shenhm/documents/lm-watermarking/watermark_reliability_release")
from watermark_processor_kwg import WatermarkDetector
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
from evaluate import load

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/shenhm/doucuments/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_4_32_0.2_LSH_v2.2_c4_new/gen_table_meta.json",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
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
from tqdm import tqdm
import math

def read_data_from_file(data_path):
    data = []
    try:
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line.strip()))  # 读取每一行并转换为字典
        print(f"Data loaded successfully from {data_path}")
    except Exception as e:
        print(f"Error while loading data from {data_path}: {e}")
    return data

import json
import os
import math
import numpy as np

def read_data_from_file(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def extract_ppl_dict(data, key='w_wm_output_log_diversity'):
    return {item['idx']: item.get(key, float('nan')) for item in data if 'idx' in item}

def compare_two_files(path1, path2, key='w_wm_output_log_diversity'):
    data1 = read_data_from_file(path1)
    data2 = read_data_from_file(path2)

    ppl_dict1 = extract_ppl_dict(data1, key)
    ppl_dict2 = extract_ppl_dict(data2, key)

    shared_indices = set(ppl_dict1.keys()) & set(ppl_dict2.keys())
    ppl_values1 = []
    ppl_values2 = []
    diffs = []

    for idx in shared_indices:
        v1 = ppl_dict1[idx]
        v2 = ppl_dict2[idx]
        if not (math.isnan(v1) or math.isnan(v2)):
            ppl_values1.append(v1)
            ppl_values2.append(v2)
            diffs.append(v1 - v2)
    print(shared_indices)
    print(f"共有 {len(shared_indices)} 个 shared idx,其中有效比较项为 {len(diffs)}")
    print(f"{path1} 平均 _log_diversity: {np.mean(ppl_values1):.4f}")
    print(f"{path2} 平均 _log_diversity: {np.mean(ppl_values2):.4f}")
    print(f"_log_diversity 平均差 (方案1 - 方案2): {np.mean(diffs):.4f}")
    print(f"_log_diversity 差值标准差: {np.std(diffs):.4f}")

if __name__ == '__main__':

    # path1 = "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/lfqa/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table_GPT.jsonl_z_score_diversity"
    # path2 = "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/lfqa/Our_test/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_16_LSH_H_16_lfqa/gen_table_GPT.jsonl_z_score_diversity"
    path1 = "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/diversity_Show/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table_safe.jsonl_diversity_20"
        # "LFQA_KGW": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/lfqa/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table_GPT.jsonl_z_score",
    path2 = "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/our_NEW_DIVERSITY_SHOW/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_16_LSH_H_16_c4/gen_table.jsonl_diversity_20"
    compare_two_files(path1, path2)
