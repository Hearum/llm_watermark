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

def main(data_path):
    args = parse_args()
    data = []

    model_id = "gpt2" #"/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
    output_file = data_path + f'_{model_id}'

    if os.path.exists(output_file):
        print('#' * 50, "result", '#' * 50)
        data = read_data_from_file(output_file)
        for key in ["w_wm_output", "w_wm_output_attacked", "no_wm_output", "baseline_completion"]:
            valid_ppls = [item.get(f"{key}_ppl", float('nan')) for item in data if not math.isnan(item.get(f"{key}_ppl", float('nan')))][:500]
            # [item.get(f"w_wm_output_ppl", float('nan')) for item in data]
            if valid_ppls:
                mean_ppl = np.mean(valid_ppls)
            else:
                mean_ppl = math.nan
            print(key, mean_ppl)
        return
    
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    perplexity = load("perplexity", module_type="metric")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

    all_texts = []
    all_keys = []

    for item in tqdm(data):
        for key in ["w_wm_output", "no_wm_output", "baseline_completion"]:
            input = item.get(key, "")
            if isinstance(input, str):
                input_text = f"{item['truncated_input']},\n,{input}"

                tokens = tokenizer.encode(input_text, truncation=True, max_length=500)
                input_text = tokenizer.decode(tokens, skip_special_tokens=True)

            if input_text and item.get(f"{key}_length", "") > 140:
                all_texts.append(input_text)
                all_keys.append((item, key))
            else:
                print(f"Skipping short input for key '{key}' in item ")
                item[f"{key}_ppl"] = math.nan

    # 一次性计算所有 PPL
    if all_texts:
        try:
            results = perplexity.compute(
                model_id=model_id,
                predictions=all_texts,
                add_start_token=False
            )
            for idx, (item, key) in enumerate(all_keys):
                if 'perplexities' in results:
                    item[f"{key}_ppl"] = round(results["perplexities"][idx], 2)
                else:
                    item[f"{key}_ppl"] = math.nan
        except (ValueError, AssertionError) as e:
            print(f"Error during PPL computation: {e}")
            for _, (item, key) in enumerate(all_keys):
                item[f"{key}_ppl"] = math.nan

    # 写回 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Updated JSONL file saved to: {output_file}")
    print('#' * 50, "result", '#' * 50)
    for key in ["w_wm_output", "w_wm_output_attacked", "no_wm_output", "baseline_completion"]:
        valid_ppls = [item.get(f"{key}_ppl", float('nan')) for item in data if not math.isnan(item.get(f"{key}_ppl", float('nan')))]
        mean_ppl = np.mean(valid_ppls) if valid_ppls else math.nan
        print(key, mean_ppl)
    return

if __name__ == '__main__':
    paths = [
            # "/home/shenhm/documents/temp/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf16-4-True-15485863/gen_table.jsonl",
            #  "/home/shenhm/documents/temp/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf32-4-True-15485863/gen_table.jsonl",
            #  "/home/shenhm/documents/temp/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf64-4-True-15485863/gen_table.jsonl",
            #  "/home/shenhm/documents/temp/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf128-4-True-15485863/gen_table.jsonl",
            "/home/shenhm/documents/temp/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table.jsonl",
             "/home/shenhm/documents/temp/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf1-4-True-15485863/gen_table.jsonl",
            #  "/home/shenhm/documents/temp/c4/PPL_KWG_TEST_NEW/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf1-3-True-15485863/gen_table.jsonl",
            #  "/home/shenhm/documents/temp/c4/PPL_KWG_TEST_NEW/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf1-2-True-15485863/gen_table.jsonl",
            #  "/home/shenhm/documents/temp/c4/PPL_KWG_TEST_NEW/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf4-4-True-15485863/gen_table.jsonl",
            # #  "/home/shenhm/documents/temp/c4/PPL_KWG_TEST_NEW/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf4-8-True-15485863/gen_table.jsonl",
            #  "/home/shenhm/documents/temp/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf16-4-True-15485863/gen_table.jsonl",
            #  "/home/shenhm/documents/temp/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf32-4-True-15485863/gen_table.jsonl"
             ]

    for path in paths:
        main(path)