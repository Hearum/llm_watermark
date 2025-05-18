
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
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/CP_attack/len_250/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table_dipper_O60_L60.jsonl",
        help="Path to the data file containing the z-scores"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/CP_attack/len_250/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table_meta.json",
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
def main():
    args = parse_args()
    with open(args.config_path, 'r', encoding='utf-8') as infile:
        config_data = json.load(infile)
    tokenizer = AutoTokenizer.from_pretrained(config_data.get('model_name_or_path'),local_files_only=True)
    
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=config_data.get('gamma'),
        seeding_scheme=config_data.get('seeding_scheme'),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        tokenizer=tokenizer,
        z_threshold=args.detection_z_threshold,
        # normalizers=args.normalizers,

        ignore_repeated_ngrams=args.ignore_repeated_ngrams,
    )
    data = []
    with open(args.data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    # 遍历数据并计算 z_score
    for item in tqdm(data):
        for key in ["w_wm_output", "w_wm_output_attacked", "no_wm_output"]:
            input_text = item.get(key, "")  # 获取文本内容
            if input_text:  # 确保文本不为空
                try:
                    # 调用 watermark_detector.detect 获取 score_dict
                    score_dict = watermark_detector.detect(
                                input_text,
                                return_prediction=False,
                                convert_to_float=True,
                            )
                    # 获取 z_score，如果没有就设置为 NaN
                    item[f"{key}_z_score"] = score_dict.get("z_score", math.nan)
                    item[f"{key}_z_score_at_T"] = score_dict.get("z_score_at_T", math.nan).tolist()
                except ValueError as e:
                    # 捕获 ValueError 异常，并将 z_score 设置为 NaN
                    if "Must have at least 1 token" in str(e):
                        item[f"{key}_z_score"] = math.nan
                    else:
                        raise  # 如果是其他 ValueError 异常，重新抛出
    # 写回 JSONL 文件
    with open(args.data_path+'_z_score_visualization', 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Updated JSONL file saved to: {args.data_path+'_z_score_visualization'}")

if __name__ == '__main__':
    main()

# import sys
# sys.path.append("/home/shenhm/documents/lm-watermarking/watermark_reliability_release")
# from watermark_processor_kwg import WatermarkDetector
# import os
# import json
# from copy import deepcopy
# from types import NoneType

# from typing import Union
# import numpy as np
# import sklearn.metrics as metrics
# import argparse  
# import torch
# from utils.submitit import str2bool  # better bool flag type for argparse
# from functools import partial
# from dataclasses import dataclass
# import os
# import json
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from torch.nn.functional import softmax
# import pdb

# import math

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--data_path",
#         type=str,
#         default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-3-True-15485863/gen_table.jsonl",
#         help="Path to the data file containing the z-scores"
#     )
#     parser.add_argument(
#         "--config_path",
#         type=str,
#         default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-3-True-15485863/gen_table_meta.json",
#     )
#     parser.add_argument(
#         "--detection_z_threshold",
#         type=float,
#         default=4.0,
#         help="The test statistic threshold for the detection hypothesis test.",
#     )
#     parser.add_argument(
#         "--ignore_repeated_ngrams",
#         type=str2bool,
#         default=False,
#         help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
#     )
#     return parser.parse_args()
# from tqdm import tqdm

# def main():

#     args = parse_args()
#     with open(args.config_path, 'r', encoding='utf-8') as infile:
#         config_data = json.load(infile)
#     tokenizer = AutoTokenizer.from_pretrained(config_data.get('model_name_or_path'),local_files_only=True)
    
#     watermark_detector = WatermarkDetector(
#         vocab=list(tokenizer.get_vocab().values()),
#         gamma=config_data.get('gamma'),
#         seeding_scheme=config_data.get('seeding_scheme'),
#         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#         tokenizer=tokenizer,
#         z_threshold=args.detection_z_threshold,
#         # normalizers=args.normalizers,
#         ignore_repeated_ngrams=args.ignore_repeated_ngrams,
#     )
#     # 读取数据文件
#     data = []
#     with open(args.data_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             data.append(json.loads(line.strip()))
#     data = data[:20]
#     # 遍历数据并计算不同截断长度下的z-score
#     for item in tqdm(data):
#         for key in ["w_wm_output", "no_wm_output"]:
#             input_text = item.get(key, "")  # 获取文本内容
#             if input_text:  # 确保文本不为空
#                 # 遍历token长度从0到150，以10为间隔
#                 for length in range(240, 250, 10):
#                     # 使用tokenizer截断文本到指定长度
#                     truncated_text = tokenizer(input_text, max_length=length, truncation=True, padding=False)
#                     decoded_text = tokenizer.decode(truncated_text['input_ids'], skip_special_tokens=True)
#                     try:
#                         # 调用 watermark_detector.detect 获取 score_dict
#                         score_dict = watermark_detector.detect(
#                             decoded_text,
#                             return_prediction=False,
#                             convert_to_float=True,
#                         )
                        
#                         # 获取 z_score，如果没有就设置为 NaN
#                         z_score = score_dict.get("z_score", math.nan)
#                         item[f"{key}_{length}_z_score"] = z_score

#                     except ValueError as e:
#                         # 捕获 ValueError 异常，并将 z_score 设置为 NaN
#                         item[f"{key}_{length}_z_score"] = math.nan

#     # 写回更新后的数据到JSONL文件
#     with open(args.data_path + '_all_len_z_score', 'w', encoding='utf-8') as file:
#         for item in data:
#             file.write(json.dumps(item, ensure_ascii=False) + "\n")
    
#     print(f"Updated JSONL file saved to: {args.data_path + '_all_len_z_score'}")

# if __name__ == '__main__':
#     main()
