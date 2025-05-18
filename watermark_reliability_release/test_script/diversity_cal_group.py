
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_32_0.2_LSH_v2.2_c4_new/gen_table.jsonl",
        help="Path to the data file containing the z-scores"
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
        "--l",
        type=int,
        default=60,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--o",
        type=int,
        default=60,
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
import argparse
import json
import nltk
import time
import os
import tqdm

from nltk.tokenize import sent_tokenize

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# nltk.download("punkt")

# from https://github.com/yxuansu/SimCTG/blob/main/simctg/evaluation.py
# as used in Contrastive Decoding https://github.com/XiangLi1999/ContrastiveDecoding
import math


def eval_text(text, ngram):
    token_list = text.strip().split()
    start_idx, end_idx = 0, ngram
    total_num = 0
    ngram_set = set()
    while end_idx < len(token_list):
        one_ngram_list = token_list[start_idx:end_idx]
        assert len(one_ngram_list) == ngram
        one_ngram = " ".join(one_ngram_list)
        total_num += 1
        ngram_set.add(one_ngram)
        start_idx += 1
        end_idx += 1
    return len(ngram_set), total_num


def eval_one_instance(text, ngram_list):
    res_dict = {}
    for n in ngram_list:
        n_unique, n_total = eval_text(text, n)
        res_dict[n] = {"unique": n_unique, "total": n_total}
    unique_token_set = set(text.strip().split())
    return res_dict, unique_token_set


def measure_repetition_and_diversity(input_text):
    """
    input text: a string
    """
    ngram_list = [2, 3, 4]
    pred_res_dict = {}
    for n in ngram_list:
        pred_res_dict[n] = {}
        pred_res_dict[n]["unique"] = 0
        pred_res_dict[n]["total"] = 0

    pred_unique_token_set = set()
    # for text in text_list:
    stripped_text = input_text.strip("\n").strip()
    one_pred_res_dict, one_pred_uni_token_set = eval_one_instance(stripped_text, ngram_list)

    # unique token set
    pred_unique_token_set = pred_unique_token_set.union(one_pred_uni_token_set)
    # ngram statistic
    for n in ngram_list:
        pred_res_dict[n]["unique"] += one_pred_res_dict[n]["unique"]
        pred_res_dict[n]["total"] += one_pred_res_dict[n]["total"]

    # prediction result
    pred_seq_2 = 1 - (pred_res_dict[2]["unique"] / pred_res_dict[2]["total"])
    # pred_seq_2 = round(pred_seq_2 * 100, 2)
    pred_seq_3 = 1 - (pred_res_dict[3]["unique"] / pred_res_dict[3]["total"])
    # pred_seq_3 = round(pred_seq_3 * 100, 2)
    pred_seq_4 = 1 - (pred_res_dict[4]["unique"] / pred_res_dict[4]["total"])
    # pred_seq_4 = round(pred_seq_4 * 100, 2)
    pred_div = (1 - pred_seq_2 / 100) * (1 - pred_seq_3 / 100) * (1 - pred_seq_4 / 100)

    pred_log_div = -math.log(max(1 - pred_div, math.exp(-20)))  # this is our addition
    # defining 20 manually as the maximal value

    # return pred_seq_2, pred_seq_3, pred_seq_4, pred_div
    # return a dictionary with the ngram repetition levels and diversity
    return {
        "repetition_2": pred_seq_2,
        "repetition_3": pred_seq_3,
        "repetition_4": pred_seq_4,
        "diversity": pred_div,
        "log_diversity": pred_log_div,
    }


dummy_rep_div_result = {
    "repetition_2": float("nan"),
    "repetition_3": float("nan"),
    "repetition_4": float("nan"),
    "diversity": float("nan"),
    "log_diversity": float("nan"),
}

import json
import os
from tqdm import tqdm

OUTPUT_TEXT_COLUMN_NAMES = [
    "baseline_completion",
    "no_wm_output",
    "w_wm_output",
    # "w_wm_output_attacked",
]

def get_processed_count(output_path):
    """获取已处理的样本数，如果没有记录则返回 0"""
    processed_file = output_path + '.processed'
    if os.path.exists(processed_file):
        with open(processed_file, 'r', encoding='utf-8') as f:
            return int(f.read().strip())
    return 0

def update_processed_count(output_path, count):
    """更新已处理的样本数"""
    processed_file = output_path + '.processed'
    with open(processed_file, 'w', encoding='utf-8') as f:
        f.write(str(count))

def compute_repetition_diversity(example, include_repetition=True, include_diversity=True):
    for col_name in OUTPUT_TEXT_COLUMN_NAMES:
        if col_name in example:
            try:
                if example[f"{col_name}_length"] ==150:
                    results_tuple = measure_repetition_and_diversity(example[col_name])
                else:
                    results_tuple = dummy_rep_div_result
            except Exception as e:
                print(
                    f"Error for '{col_name}' computing repetition and diversity on text: '{example[col_name]}'\nError:{e}"
                )
                results_tuple = dummy_rep_div_result

            if include_repetition:
                # returns pred_seq_2, pred_seq_3, pred_seq_4, pred_div
                # add each key from the result tuple to the example, prepending the col_name
                metrics_dict = {f"{col_name}_{key}": value for key, value in results_tuple.items()}
                example.update(metrics_dict)
            if include_diversity:
                # returns diversity only
                example[f"{col_name}_diversity"] = results_tuple["diversity"]
                example[f"{col_name}_log_diversity"] = results_tuple["log_diversity"]
    return example

def convert_item(item):
    for key, value in item.items():
        if isinstance(value, np.float32):  # 检查是否为 float32 类型
            item[key] = float(value)  # 转换为 Python 原生的 float
    return item

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

    output_file = data_path + '_diversity_20=150'

    if os.path.exists(output_file):
        print('#' * 50, "result", '#' * 50)
        data = read_data_from_file(output_file)
        for key in ["w_wm_output", "no_wm_output"]:
            valid_ppls = [item.get(f"{key}_log_diversity", float('nan')) for item in data if not math.isnan(item.get(f"{key}_log_diversity", float('nan')))]
            mean_log_diversity = np.mean(valid_ppls) if valid_ppls else math.nan
            print(key, mean_log_diversity)
            # valid_ppls = [item.get(f"{key}_diversity", float('nan')) for item in data if not math.isnan(item.get(f"{key}_diversity", float('nan')))]
            # mean_diversity = np.mean(valid_ppls) if valid_ppls else math.nan
            # print(key, mean_diversity)
        key = "w_wm_output"
        valid_ppls = [item.get(f"{key}_log_diversity", float('nan')) for item in data if not math.isnan(item.get(f"{key}_log_diversity", float('nan')))]
        pdb.set_trace()
        mean_log_diversity1 = np.mean(valid_ppls) if valid_ppls else math.nan

        key = "no_wm_output"
        valid_ppls = [item.get(f"{key}_log_diversity", float('nan')) for item in data if not math.isnan(item.get(f"{key}_log_diversity", float('nan')))]
        mean_log_diversity2 = np.mean(valid_ppls) if valid_ppls else math.nan
        pdb.set_trace()
        print("rate:",mean_log_diversity1/mean_log_diversity2)

        return
    

    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    
    for item in tqdm(data):
        item = compute_repetition_diversity(item)

    # 写回 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in data:
            item = convert_item(item)
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Updated JSONL file saved to: {output_file}")

    # 打印结果

    for key in ["w_wm_output", "no_wm_output","baseline_completion"]:
        valid_ppls = [item.get(f"{key}_log_diversity", float('nan')) for item in data if not math.isnan(item.get(f"{key}_log_diversity", float('nan')))]
        mean_log_diversity = np.mean(valid_ppls) if valid_ppls else math.nan
        print(key, mean_log_diversity)
        valid_ppls = [item.get(f"{key}_diversity", float('nan')) for item in data if not math.isnan(item.get(f"{key}_diversity", float('nan')))]
        mean_diversity = np.mean(valid_ppls) if valid_ppls else math.nan
        print(key, mean_diversity)

    # if psp_values:
    #     mean_psp = np.mean(psp_values)
    # else:
    #     mean_psp = math.nan
    # print("Mean wm_and_no_wm_psp:", mean_psp)

if __name__ == '__main__':

    paths = {
        # "WikiText_Ours": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/Ours_fin/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_16_LSH_H_16_wikitext/gen_table_GPT.jsonl_z_score",
        "C4_Ours": "/home/shenhm/documents/temp/c4/Our_test/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_16_LSH_H_16_c4/gen_table_GPT.jsonl_z_score",
        # "C4-4-4":"/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/KWG_TEST/len_250/llama_7B_N500_T200_no_filter_batch_1_delta_4_gamma_0.25_KWG_ff-anchored_minhash_prf4-4-True-15485863/gen_table_GPT.jsonl_z_score",
        #"LFQA_Ours": "/home/shenhm/documents/temp/debug_temp_file/new_test_Ours/NEWS_ours_plus_1/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_5_LSH_H_6_c4/gen_table.jsonl",
        # # "WikiText_Ours2": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/Ours_fin/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_LSH_H_6_wikitext/gen_table_GPT.jsonl_z_score",
        # # "C4_Ours2": "/home/shenhm/documents/temp/c4/Our_test/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_LSH_H_6_c4/gen_table_GPT.jsonl_z_score",
        # # "LFQA_Ours2": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/lfqa/Our_test/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_LSH_H_6_lfqa/gen_table_GPT.jsonl_z_score",
        #"WikiText_KGW": "/home/shenhm/documents/temp/debug_temp_file/new_test_Ours/NEWS_ours_plus_1/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_12_LSH_H_16_c4/gen_table.jsonl",
        # "C4_KGW": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/Ours_4.0_test/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_LSH_H_6_c4/gen_table.jsonl",
        # "LFQA_KGW": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/lfqa/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table_GPT.jsonl_z_score",
        #"text3":"/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/our_NEW_DIVERSITY_SHOW/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_16_LSH_H_16_c4/gen_table.jsonl"
        # "text1":"/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/Ours_fin/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_16_LSH_H_16_wikitext/gen_table_GPT.jsonl_z_score",
        # "text2":"/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/CP_attack/len_250/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table.jsonl",
        # "text3":"/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/diversity_Show/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table.jsonl",
        # "text4":"/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/diversity_Show/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table.jsonl",
    }
    # paths = [
    #         "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/Ours_fin/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_LSH_H_6_wikitext/gen_table_GPT.jsonl_z_score",
    #         "/home/shenhm/documents/temp/c4/Our_test/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_LSH_H_6_c4/gen_table_GPT.jsonl_z_score",
    #         "/home/shenhm/doucuments/lm-watermarking/watermark_reliability_release/output/c4/PPL_KWG_TEST_NEW/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf4-4-True-15485863/gen_table.jsonl",
    #         "/home/shenhm/doucuments/lm-watermarking/watermark_reliability_release/output/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf16-4-True-15485863/gen_table.jsonl",
    #         "/home/shenhm/doucuments/lm-watermarking/watermark_reliability_release/output/c4/MIN_KWG_TEST_NEW_DEMO/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-minhash_prf16-4-True-15485863/gen_table.jsonl",
    #         "/home/shenhm/doucuments/lm-watermarking/watermark_reliability_release/output/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf32-4-True-15485863/gen_table.jsonl",
    #         "/home/shenhm/doucuments/lm-watermarking/watermark_reliability_release/output/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf64-4-True-15485863/gen_table.jsonl",
    #         "/home/shenhm/doucuments/lm-watermarking/watermark_reliability_release/output/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf128-4-True-15485863/gen_table.jsonl",
    #         # "/home/shenhm/doucuments/lm-watermarking/watermark_reliability_release/output/c4/MIN_KWG_TEST_NEW_DEMO/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-minhash_prf512-4-True-15485863/gen_table.jsonl",
    #         # "/home/shenhm/doucuments/lm-watermarking/watermark_reliability_release/output/c4/MIN_KWG_TEST_NEW_DEMO/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-minhash_prf1024-4-True-15485863/gen_table.jsonl",
            
    #         # "/home/shenhm/doucuments/lm-watermarking/watermark_reliability_release/output/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table_GPT.jsonl_z_score",
    #         # "/home/shenhm/doucuments/lm-watermarking/watermark_reliability_release/output/c4/PPL_KWG_TEST_NEW/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf1-3-True-15485863/gen_table_GPT.jsonl_z_score",
    #         #  "/home/shenhm/documents/temp/c4/PPL_KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf1-4-True-15485863/gen_table.jsonl",
    #     ]

    for path in paths.values():
        main(path)
