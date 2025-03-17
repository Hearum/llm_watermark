
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
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_4_32_0.2_LSH_v2.2_c4_new/gen_table_meta.json",
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


def main(data_path):
    
    args = parse_args()
    data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    perplexity = load("perplexity", module_type="metric")
    tokenizer = AutoTokenizer.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9", local_files_only=True)

    batch_size = 32  # 可以根据实际内存和计算能力调整批次大小
    batch_data = []  # 用于存储当前批次的文本
    batch_keys = []  # 用于存储对应文本的key（用于后续赋值）

    for item in tqdm(data):
        for key in ["w_wm_output", "w_wm_output_attacked", "no_wm_output", "baseline_completion"]:
            input_text = item.get(key, "")
            if isinstance(input_text, str):
                # 对输入文本进行编码并限制最大长度为160
                tokens = tokenizer.encode(input_text, truncation=True, max_length=160)
                input_text = tokenizer.decode(tokens, skip_special_tokens=True)
            else:
                # 如果不是字符串，处理为默认值
                print(f"Warning: Expected a string for key '{key}', but got {type(input_text)}")
                input_text = ""  
                tokens = []

            # 如果 `input_text` 非空并且 tokens 长度大于1，则加入 batch_data
            if input_text and len(tokens) > 10:
                    batch_data.append(input_text)
                    batch_keys.append((item, key))
            else:
                # 处理短文本或空文本的情况
                print(f"Skipping short input for key '{key}' in item (token length: {len(tokens)})")

            # 如果当前批次已经达到设定的大小，则计算困惑度
            if len(batch_data) >= batch_size:
                try:
                    results = perplexity.compute(model_id='gpt2', add_start_token=False, predictions=batch_data)
                    for idx, (item, key) in enumerate(batch_keys):
                        if 'perplexities' in results:
                            item[f"{key}_ppl"] = round(results["perplexities"][idx], 2)
                        else:
                            item[f"{key}_ppl"] = math.nan
                except (ValueError,AssertionError) as e:
                    if "Must have at least 1 token" in str(e):
                        for idx, (item, key) in enumerate(batch_keys):
                            item[f"{key}_ppl"] = math.nan
                    else:
                        raise
                batch_data = []
                batch_keys = []

    # 处理剩余的文本（如果有）
    if batch_data:
        try:
            results = perplexity.compute(model_id='gpt2', add_start_token=False, predictions=batch_data)
            for idx, (item, key) in enumerate(batch_keys):
                if 'perplexities' in results:
                    item[f"{key}_ppl"] = round(results["perplexities"][idx], 2)
                else:
                    item[f"{key}_ppl"] = math.nan
        except ValueError as e:
            if "Must have at least 1 token" in str(e):
                for idx, (item, key) in enumerate(batch_keys):
                    item[f"{key}_ppl"] = math.nan
            else:
                raise

    # 写回 JSONL 文件
    with open(data_path+'_ppl', 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Updated JSONL file saved to: {data_path + '_ppl'}")
    # 打印结果
    print('#' * 50, "result", '#' * 50)
    for key in ["w_wm_output", "w_wm_output_attacked", "no_wm_output", "baseline_completion"]:
        valid_ppls = [item.get(f"{key}_ppl", float('nan')) for item in data if not math.isnan(item.get(f"{key}_ppl", float('nan')))]
        if valid_ppls:
            mean_ppl = np.mean(valid_ppls)
        else:
            mean_ppl = math.nan
        print(key, mean_ppl)
if __name__ == '__main__':
    paths = ["/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/delta5_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_selfhash_wikit/gen_table_GPT.jsonl_z_score",]

    for path in paths:
        main(path)