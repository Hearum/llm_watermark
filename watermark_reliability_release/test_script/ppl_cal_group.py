
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
    model_id = '/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9' #'facebook/opt-2.7b'#"facebook/opt-2.7b"

    output_file = data_path + f'_{"llama7b"}_ppl140'

    if os.path.exists(output_file):
        print('#' * 50, "result", '#' * 50)
        data = read_data_from_file(output_file)
        for key in ["w_wm_output", "w_wm_output_attacked", "no_wm_output", "baseline_completion"]:
            valid_ppls = [item.get(f"{key}_ppl", float('nan')) for item in data if not math.isnan(item.get(f"{key}_ppl", float('nan')))]
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
            input_text = item.get(key, "")
            if isinstance(input_text, str):
                tokens = tokenizer.encode(input_text)[:150]
                input_text = tokenizer.decode(tokens)
            else:
                print(f"Warning: Expected a string for key '{key}', but got {type(input_text)}")
                input_text = ""
                tokens = []

            if input_text and len(tokens) > 140:
                all_texts.append(input_text)
                all_keys.append((item, key))
            else:
                print(f"Skipping short input for key '{key}' in item (token length: {len(tokens)})")
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
    # paths = {
    #     # "WikiText_Ours_7b": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/Ours_fin/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_LSH_H_6_wikitext/gen_table_GPT.jsonl_z_score_gpt2_ppl140",
    #     # "C4_Ours_7b": "/home/shenhm/documents/temp/c4/Our_test/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_LSH_H_6_c4/gen_table_GPT.jsonl_z_score_gpt2_ppl140",
    #     # "LFQA_Ours_7b": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/lfqa/Our_test/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_LSH_H_6_lfqa/gen_table_GPT.jsonl_z_score_gpt2_ppl140",
    #     # "WikiText_KGW_7b": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/delta5_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_selfhash_wikit/gen_table_GPT.jsonl_ppl",
    #     # "C4_KGW_7b": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_self_wiki_c4_new/gen_table_GPT.jsonl_ppl",
    #     # "LFQA_KGW_7b": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_self_wiki_c4_new/gen_table_GPT.jsonl_ppl",

    #     "WikiText_Ours_13": "/home/shenhm/documents/temp/PPL_visualization/wikitext/gen_table.jsonl",
    #     "C4_Ours_13": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/len_150/llama_13B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_32_0.25_LSH_v2.2_c4_new/gen_table.jsonl",
    #     "LFQA_Ours_13": "/home/shenhm/documents/temp/PPL_visualization/lfqa/gen_table.jsonl",
        
    #     "WikiText_KGW_13": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/ppl_Show/len_150/llama_13B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table.jsonl",
    #     "C4_KGW_13": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/len_150/llama_13B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table.jsonl",
    #     "lfqa_KGW_13": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/lfqa/ppl_Show/len_250/llama_13B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table.jsonl",
    #     # "text2":"/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/KWG_TEST/len_250/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-32-True-15485863/gen_table.jsonl",
    #     # "text3":"/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/KWG_TEST/len_250/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf4-32-True-15485863/gen_table.jsonl",
    #     # "text3":"/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/KWG_TEST/len_250/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf4-32-True-15485863/gen_table.jsonl",
    #     # "C4_KGW": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_self_wiki_c4_new/gen_table_GPT.jsonl_z_score",
    #     # "LFQA_KGW": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/lfqa/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table_GPT.jsonl_z_score",
    #     # "text1":"/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/CP_attack/len_250/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_LSH_H_6_c4/gen_table.jsonl",
    #     # "text2":"/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/CP_attack/len_250/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table.jsonl"
    # }
    paths = {
        "WikiText_Ours": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/Ours_fin/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_LSH_H_6_wikitext/gen_table_GPT.jsonl_z_score_gpt2_ppl140",
        "C4_Ours": "/home/shenhm/documents/temp/c4/Our_test/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_LSH_H_6_c4/gen_table_GPT.jsonl_z_score_gpt2_ppl140",
        "LFQA_Ours": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/lfqa/Our_test/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_LSH_H_6_lfqa/gen_table_GPT.jsonl_z_score_gpt2_ppl140",
        "WikiText_KGW": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/delta5_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_selfhash_wikit/gen_table_GPT.jsonl_ppl",
        "C4_KGW": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_self_wiki_c4_new/gen_table_GPT.jsonl_ppl",
        "LFQA_KGW": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_self_wiki_c4_new/gen_table_GPT.jsonl_ppl",
        
        "WikiText_Ours1": "/home/shenhm/documents/temp/PPL_visualization/wikitext/gen_table.jsonl_gpt2_ppl140",
        "C4_Ours1": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/len_150/llama_13B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_32_0.25_LSH_v2.2_c4_new/gen_table.jsonl_gpt2_ppl140",
        "LFQA_Ours1": "/home/shenhm/documents/temp/PPL_visualization/lfqa/gen_table.jsonl_gpt2_ppl140",
        "WikiText_KGW1": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/ppl_Show/len_150/llama_13B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table.jsonl_gpt2_ppl140",
        "C4_KGW1": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/len_150/llama_13B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table.jsonl_gpt2_ppl140",
        "LFQA_KGW1": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/lfqa/ppl_Show/len_250/llama_13B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table.jsonl_gpt2_ppl140",
    }

    for path in paths.values():
        main(path)