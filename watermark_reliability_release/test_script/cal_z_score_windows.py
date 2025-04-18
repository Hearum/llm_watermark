
import sys
sys.path.append("/home/shenhm/documents/lm-watermarking/watermark_reliability_release")
from wateramrk_processor_windows import WatermarkDetector

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/windows_text/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6__0.25_LSH_H_6_c4/gen_table_deepseek_attacker.jsonl",
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
    # parser.add_argument(
    #     "--n_features",
    #     type=int,
    #     default=32,
    # )    
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
import pdb

def main():
    args = parse_args()
    # with open(args.config_path, 'r', encoding='utf-8') as infile:
    #     config_data = json.load(infile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",local_files_only=True)
    with open(args.config_path, 'r', encoding='utf-8') as infile:
        config_data = json.load(infile)
    # watermark_detector = WatermarkDetector(
    #     gamma=args.gamma,
    #     delta=args.delta,
    #     device=device,
    #     tokenizer=tokenizer,
    #     vocab=list(tokenizer.get_vocab().values()),
    #     z_threshold=4.0,
    #     normalizers=["unicode"],
    #     ignore_repeated_ngrams=False,
    #     threshold_len=16,
    #     windows_h_uesd  = True
    # )
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=config_data.get('gamma'),
        seeding_scheme=config_data.get('seeding_scheme'),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        tokenizer=tokenizer,
        z_threshold=args.detection_z_threshold,
        # normalizers=args.normalizers,
        ignore_repeated_ngrams=args.ignore_repeated_ngrams,
        n_hashes = config_data.get('n_hashes'),               # LSH的哈希函数数量，决定了有多少个桶
        # n_features=config_data.get('n_features'),            # 每个哈希函数的维度
        threshold=config_data.get('threshold'),
        visualization=True,
        threshold_len=config_data.get('h_win'),
        windows_h_uesd  = True
    )
    data = []
    with open(args.data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    # 遍历数据并计算 z_score
    for item in tqdm(data):
        for key in ["w_wm_output", "w_wm_output_attacked", "no_wm_output",]:
            input_text = item.get(key, "")  # 获取文本内容
            if input_text:  # 确保文本不为空
                # input_text = input_text[:2000]
                try:
                    # 调用 watermark_detector.detect 获取 score_dict
                    score_dict = watermark_detector.detect(
                                input_text,
                                return_prediction=False,
                                convert_to_float=True,
                            )
                    # 获取 z_score，如果没有就设置为 NaN
                    item[f"{key}_z_score"] = score_dict.get("z_score", math.nan)
                    
                    # 获取 activated 信息并避免 ZeroDivisionError
                    info_list = score_dict.get('info', [])
                    all_activated = [item['activated'] for item in info_list if item is not None]
                    
                    # 确保 all_activated 不是空列表，避免除零错误
                    if all_activated:
                        item[f"activated_rate"] = sum(all_activated) / len(all_activated)
                    else:
                        item[f"activated_rate"] = math.nan  # 如果没有激活信息，设置为 NaN

                    # 获取 input_ids 的长度并计算 advantage_token_protect
                    len_input_ids = [len(item['input_ids']) for item in info_list if item is not None]
                    
                    # 确保 len_input_ids 不是空列表，避免除零错误
                    if len_input_ids and len(all_activated):
                        item[f"{key}_advantage_token_protect"] = sum(len_input_ids) / len(all_activated)
                    else:
                        item[f"{key}_advantage_token_protect"] = math.nan  # 如果没有数据，设置为 NaN

                    # 计算 no_activated_rate
                    item[f"{key}_no_activated_rate"] = 1 - item.get(f"activated_rate", math.nan)
                    
                except ValueError as e:
                    # 捕获 ValueError 异常，并将 z_score 设置为 NaN
                    if "Must have at least 1 token" in str(e):
                        item[f"{key}_z_score"] = math.nan
                    else:
                        raise  # 如果是其他 ValueError 异常，重新抛出
                except ValueError as e:
                    # 捕获 ValueError 异常，并将 z_score 设置为 NaN
                    if "Must have at least 1 token" in str(e):
                        item[f"{key}_z_score"] = math.nan
                    else:
                        raise  # 如果是其他 ValueError 异常，重新抛出
    # 写回 JSONL 文件
    with open(args.data_path+'_z_score', 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Updated JSONL file saved to: {args.data_path+'_z_score'}")

if __name__ == '__main__':
    main()