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
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_self_wiki_c4_new/gen_table.jsonl",
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
    parser.add_argument("--cp_attack_type", 
                        type=str, 
                        default="k-t", 
                        choices=["single-single", "triple-single", "k-t"],
                        help="Type of copy-paste attack to apply")
    
    parser.add_argument("--cp_attack_insertion_len", type=int, default=30,
                        help="Length of each inserted fragment")
    # 插入片段数量（k），仅用于 k-t 攻击
    parser.add_argument("--cp_attack_num_insertions", type=int, default=3,
                        help="Number of insertions to perform (only for k-t attack)")
    # 最小 token 数要求，用于判断样本是否足够长才能攻击
    parser.add_argument("--cp_attack_min_len", type=int, default=150,
                        help="Minimum token count required for an example to be attacked")
    parser.add_argument("--verbose", action="store_true",
                        help="Whether to print detailed debug info during attack")
    
    return parser.parse_args()
import torch

OUTPUT_TEXT_COLUMN_NAMES = [
    # "baseline_completion",
    "no_wm_output",
    "w_wm_output",
    # "w_wm_output_attacked",
]

# 单次插入攻击
def single_insertion(
    attack_len,
    min_token_count,
    tokenized_no_wm_output,  # dst
    tokenized_w_wm_output,  # src
):
    top_insert_loc = min_token_count - attack_len
    rand_insert_locs = torch.randint(low=0, high=top_insert_loc, size=(2,))

    # tokenized_no_wm_output_cloned = torch.clone(tokenized_no_wm_output) # used to be tensor
    tokenized_no_wm_output_cloned = torch.tensor(tokenized_no_wm_output)
    tokenized_w_wm_output = torch.tensor(tokenized_w_wm_output)

    tokenized_no_wm_output_cloned[
        rand_insert_locs[0].item() : rand_insert_locs[0].item() + attack_len
    ] = tokenized_w_wm_output[rand_insert_locs[1].item() : rand_insert_locs[1].item() + attack_len]

    return tokenized_no_wm_output_cloned

# 三次插入攻击（等长）
def triple_insertion_single_len(
    attack_len,
    min_token_count,
    tokenized_no_wm_output,  # dst
    tokenized_w_wm_output,  # src
):
    tmp_attack_lens = (attack_len, attack_len, attack_len)

    while True:
        rand_insert_locs = torch.randint(low=0, high=min_token_count, size=(len(tmp_attack_lens),))
        _, indices = torch.sort(rand_insert_locs)

        if (
            rand_insert_locs[indices[0]] + attack_len <= rand_insert_locs[indices[1]]
            and rand_insert_locs[indices[1]] + attack_len <= rand_insert_locs[indices[2]]
            and rand_insert_locs[indices[2]] + attack_len <= min_token_count
        ):
            break

    # replace watermarked sections into unwatermarked ones
    # tokenized_no_wm_output_cloned = torch.clone(tokenized_no_wm_output) # used to be tensor
    tokenized_no_wm_output_cloned = torch.tensor(tokenized_no_wm_output)
    tokenized_w_wm_output = torch.tensor(tokenized_w_wm_output)

    for i in range(len(tmp_attack_lens)):
        start_idx = rand_insert_locs[indices[i]]
        end_idx = rand_insert_locs[indices[i]] + attack_len

        tokenized_no_wm_output_cloned[start_idx:end_idx] = tokenized_w_wm_output[start_idx:end_idx]

    return tokenized_no_wm_output_cloned

# K 次插入攻击（通用）
def k_insertion_t_len(
    num_insertions,
    insertion_len,
    min_token_count,
    tokenized_dst_output,  # dst
    tokenized_src_output,  # src
    verbose=False,
):
    insertion_lengths = [insertion_len] * num_insertions

    # these aren't save to rely on indiv, need to use the min of both
    # dst_length = len(tokenized_dst_output)
    # src_length = len(tokenized_src_output) # not needed, on account of considering only min_token_count
    # as the max allowed index

    while True:
        rand_insert_locs = torch.randint(
            low=0, high=min_token_count, size=(len(insertion_lengths),)
        )
        _, indices = torch.sort(rand_insert_locs)

        if verbose:
            print(
                f"indices: {[rand_insert_locs[indices[i]] for i in range(len(insertion_lengths))]}"
            )
            print(
                f"gaps: {[rand_insert_locs[indices[i + 1]] - rand_insert_locs[indices[i]] for i in range(len(insertion_lengths) - 1)] + [min_token_count - rand_insert_locs[indices[-1]]]}"
            )

        # check for overlap condition for all insertions
        overlap = False
        for i in range(len(insertion_lengths) - 1):
            if (
                rand_insert_locs[indices[i]] + insertion_lengths[indices[i]]
                > rand_insert_locs[indices[i + 1]]
            ):
                overlap = True
                break

        if (
            not overlap
            and rand_insert_locs[indices[-1]] + insertion_lengths[indices[-1]] < min_token_count
        ):
            break

    # replace watermarked sections into unwatermarked ones

    tokenized_dst_output_cloned = torch.tensor(tokenized_dst_output)
    tokenized_src_output = torch.tensor(tokenized_src_output)

    for i in range(len(insertion_lengths)):
        start_idx = rand_insert_locs[indices[i]]
        end_idx = rand_insert_locs[indices[i]] + insertion_lengths[indices[i]]

        tokenized_dst_output_cloned[start_idx:end_idx] = tokenized_src_output[start_idx:end_idx]

    return tokenized_dst_output_cloned


import argparse
import json
import nltk
import time
import os
import tqdm

from nltk.tokenize import sent_tokenize

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
def check_output_column_lengths(example, min_len=0):
    baseline_completion_len = example["baseline_completion_length"]
    no_wm_output_len = example["no_wm_output_length"]
    w_wm_output_len = example["w_wm_output_length"]
    conds = all(
        [
            baseline_completion_len >= min_len,
            no_wm_output_len >= min_len,
            w_wm_output_len >= min_len,
        ]
    )
    return conds

# 遍历所有输出列（OUTPUT_TEXT_COLUMN_NAMES，如 w_wm_output, no_wm_output 等）
# 用 tokenizer 编码成 token 序列，存储在新字段如 w_wm_output_tokd 中。
def tokenize_for_copy_paste(example, tokenizer=None, args=None):
    for text_col in OUTPUT_TEXT_COLUMN_NAMES:
        if text_col in example:
            example[f"{text_col}_tokd"] = tokenizer(example[text_col], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            # example[text_col] tokenizer.encode(example[text_col], return_tensors="pt")
    return example


def copy_paste_attack(example, tokenizer, args=None):
    # check if the example is long enough to attack
    assert tokenizer is not None, "tokenizer must be provided"
    example = tokenize_for_copy_paste(example=example,tokenizer=tokenizer)

    # else, attack
    # Understanding the functionality:
    # we always write the result into the "w_wm_output_attacked" column
    # however depending on the detection method we're targeting, the
    # "src" and "dst" columns will be different. However,
    # the internal logic for these functions has old naming conventions of
    # watermarked always being the insertion src and no_watermark always being the dst
    """"
    始终默认 watermarked 是插入源 source
    no_watermark 是插入目标 destination
    尽管最终的攻击结果统一写入 w_wm_output_attacked 字段，但在攻击过程中，实际的“插入源”和“插入目标”是根据具体测试策略动态设置的。
    而内部实现默认认为：带水印文本是“插入源（src）”，不带水印文本是“目标文本（dst）”，这一点在使用时需要注意。
    """
    # 准备 token 序列
    tokenized_dst = example["no_wm_output_tokd"]
    tokenized_src = example["w_wm_output_tokd"]
    min_token_count = min(len(tokenized_dst), len(tokenized_src))
   
    # 	插入一个片段
    if args.cp_attack_type == "single-single":  # 1-t
        tokenized_attacked_output = single_insertion(
            args.cp_attack_insertion_len,
            min_token_count,
            tokenized_dst,
            tokenized_src,
        )
    # 插入三个非重叠片段（等长）
    elif args.cp_attack_type == "triple-single":  # 3-t
        tokenized_attacked_output = triple_insertion_single_len(
            args.cp_attack_insertion_len,
            min_token_count,
            tokenized_dst,
            tokenized_src,
        )
    # 插入 k 个固定长度 t 的片段
    elif args.cp_attack_type == "k-t":
        tokenized_attacked_output = k_insertion_t_len(
            args.cp_attack_num_insertions,  # k
            args.cp_attack_insertion_len,  # t
            min_token_count,
            tokenized_dst,
            tokenized_src,
            verbose=args.verbose,
        )
    elif args.cp_attack_type == "k-random":  # k-t | k>=3, t in [floor(T/2k), T/k)
        raise NotImplementedError(f"Attack type {args.cp_attack_type} not implemented")
    elif args.cp_attack_type == "triple-triple":  # 3-(k_1,k_2,k_3)
        raise NotImplementedError(f"Attack type {args.cp_attack_type} not implemented")
    else:
        raise ValueError(f"Invalid attack type: {args.cp_attack_type}")

    example["w_wm_output_attacked"] = tokenizer.batch_decode(
        [tokenized_attacked_output], skip_special_tokens=True
    )[0]
    example["w_wm_output_attacked_length"] = len(tokenized_attacked_output)

    return example


import json
import os
from tqdm import tqdm


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

def main():
    args = parse_args()

    input_path = args.data_path
    output_path = os.path.splitext(input_path)[0] + f'_{args.cp_attack_type}_t_{args.cp_attack_insertion_len}_k_{args.cp_attack_num_insertions}.jsonl'  # 修改输出路径
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", local_files_only=True)
    # 获取上次处理到的行数
    processed_count = get_processed_count(output_path)

    total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
    with open(input_path, 'r', encoding='utf-8') as infile, \
        open(output_path, 'a', encoding='utf-8') as outfile:
        for i, line in enumerate(tqdm(infile, desc="Processing samples", initial=processed_count, total=total_lines)):
            # if i < processed_count:  # 如果当前行已经处理过，则跳过
            #     print(i)
            #     continue
            data_item = json.loads(line.strip())  # 读取并解析每一行数据
            if data_item["w_wm_output_length"] < 50:
                print(data_item["w_wm_output_length"],"is too short, pass")
                continue
            if check_output_column_lengths(data_item, min_len=args.cp_attack_min_len):
                updated_item = copy_paste_attack(example=data_item,tokenizer=tokenizer,args=args) 
            else:
                continue

            for col in OUTPUT_TEXT_COLUMN_NAMES:
                key = f"{col}_tokd"
                if key in updated_item:
                    del updated_item[key]
            outfile.write(json.dumps(updated_item, ensure_ascii=False) + "\n")
    
            # 每处理一个样本，更新已处理行数
            processed_count += 1
            update_processed_count(output_path, processed_count)

    print(f"Updated JSONL file saved to: {output_path}")

if __name__ == '__main__':
    main()
