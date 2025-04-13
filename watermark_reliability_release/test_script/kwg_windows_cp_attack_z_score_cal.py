
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
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_self_wiki_c4_new/gen_table_k-t_t_30_k_3.jsonl",
        help="Path to the data file containing the z-scores"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_self_wiki_c4_new/gen_table_meta.json",
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
    parser.add_argument(
        "--window_settings",
        type=str,
        default="20,40,max",  # can also be "20" or "20,40,max"
        help="Comma separated list of window sizes to use for watermark detection. Only used if 'windowed-z-score' is in the evaluation metrics list.",
    )
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

def safe_convert(v):
    if isinstance(v, (torch.Tensor, np.ndarray)):
        return v.tolist()
    elif isinstance(v, (float, int, str, list, dict, type(None))):
        return v
    else:
        return str(v)
from tqdm import tqdm
import math

def load_z_scores(file_path,key):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    # Assuming 'w_wm_output_z_score' and 'no_wm_output_z_score' are the relevant keys
    human_z_scores = [entry.get(f'no_wm_output_win{key}-1_z_score', math.nan) for entry in data]
    # w_wm_output_attacked_z_score  w_wm_output_z_score
    machine_z_scores = [entry.get(f'w_wm_output_attacked_win{key}-1_z_score', math.nan) for entry in data]
    # pdb.set_trace()
    # np.nanmean(machine_z_scores)
    # np.nanmean(human_z_scores)
    # np.nanmax(machine_z_scores)
    # np.nanmax(human_z_scores)
    # sum(machine_z_scores) / len(machine_z_scores)
    return human_z_scores, machine_z_scores

def load_z_scores_2(file_path,key):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    # Assuming 'w_wm_output_z_score' and 'no_wm_output_z_score' are the relevant keys
    human_z_scores = [entry.get(f'no_wm_output_win{key}-1_z_score', math.nan) for entry in data]
    # w_wm_output_attacked_z_score  w_wm_output_z_score
    machine_z_scores = [entry.get(f'w_wm_output_win{key}-1_z_score', math.nan) for entry in data]

    return human_z_scores, machine_z_scores


def get_roc_auc(human_z, machine_z):
    assert len(human_z) == len(machine_z)

    # Combine human and machine z-scores
    all_scores = np.concatenate([np.array(human_z), np.array(machine_z)])
    
    # Generate labels for the scores (0 for human, 1 for machine)
    baseline_labels = np.zeros_like(human_z)
    watermark_labels = np.ones_like(machine_z)
    all_labels = np.concatenate([baseline_labels, watermark_labels])

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc, fpr, tpr, thresholds
def clean_z_scores(human_z, machine_z):
    # Filter out any NaN values from both human_z and machine_z
    valid_human_z = [z for z in human_z if not np.isnan(z)]
    valid_machine_z = [z for z in machine_z if not np.isnan(z)]
    
    # Ensure both lists are of the same length
    min_len = min(len(valid_human_z), len(valid_machine_z))
    return valid_human_z[:min_len], valid_machine_z[:min_len]

def get_tpr(fpr, tpr, error_rate):
    assert len(fpr) == len(tpr)

    value = None
    for f, t in zip(fpr, tpr):
        if f <= error_rate:
            value = t
        else:
            assert value is not None
            return value

    assert value == 1.0
    return value

def AUROC(data_path,args):

    # 结果保存文件路径
    for key in args.window_settings:
        result_file_path = os.path.splitext(data_path)[0] + f"_win{key}_aucroc_result"
        # result_file_path = os.path.join(
        #     os.path.dirname(args.da),
        #     f"{os.path.basename(output_path).replace('.jsonl', '')}_win{window_size}_aucroc_result.txt"
        # )
        # Open the result file for writing
        with open(result_file_path, 'w') as result_file:
            # Write the header for the results
            result_file.write(f"Results for {data_path}\n")
            result_file.write("=" * 50 + "\n")

            # First run
            print("Processing first z-scores...")

            human_z, machine_z = load_z_scores(data_path,key)
            # Clean NaN values from z-scores
            human_z, machine_z = clean_z_scores(human_z, machine_z)

            # Calculate AUC-ROC and TPR values
            roc_auc, fpr, tpr, _ = get_roc_auc(human_z, machine_z)
            print(f"ROC AUC: {roc_auc}")
            result_file.write(f"ROC AUC: {roc_auc}\n")

            # TPR (FPR = 0%)
            tpr_value0 = get_tpr(fpr, tpr, 0.0)
            print(f"TPR (FPR = 0%): {tpr_value0}")
            result_file.write(f"TPR (FPR = 0%): {tpr_value0}\n")

            # TPR (FPR = 1%)
            tpr_value1 = get_tpr(fpr, tpr, 0.01)
            print(f"TPR (FPR = 1%): {tpr_value1}")
            result_file.write(f"TPR (FPR = 1%): {tpr_value1}\n")

            # TPR (FPR = 5%)
            tpr_value5 = get_tpr(fpr, tpr, 0.05)
            print(f"TPR (FPR = 5%): {tpr_value5}")
            result_file.write(f"TPR (FPR = 5%): {tpr_value5}\n")
            result_file.write('#' * 50 + "\n")

            # # Second run
            # print("Processing second z-scores...")
            # human_z, machine_z = load_z_scores_2(data_path,key)

            # # Clean NaN values from z-scores
            # human_z, machine_z = clean_z_scores(human_z, machine_z)
            # pdb.set_trace()
            # # Calculate AUC-ROC and TPR values
            # roc_auc, fpr, tpr, _ = get_roc_auc(human_z, machine_z)
            # print(f"ROC AUC: {roc_auc}")
            # result_file.write(f"ROC AUC: {roc_auc}\n")

            # # TPR (FPR = 0%)
            # tpr_value0 = get_tpr(fpr, tpr, 0.0)
            # print(f"TPR (FPR = 0%): {tpr_value0}")
            # result_file.write(f"TPR (FPR = 0%): {tpr_value0}\n")

            # # TPR (FPR = 1%)
            # tpr_value1 = get_tpr(fpr, tpr, 0.01)
            # print(f"TPR (FPR = 1%): {tpr_value1}")
            # result_file.write(f"TPR (FPR = 1%): {tpr_value1}\n")

            # # TPR (FPR = 5%)
            # tpr_value5 = get_tpr(fpr, tpr, 0.05)
            # print(f"TPR (FPR = 5%): {tpr_value5}")
            # result_file.write(f"TPR (FPR = 5%): {tpr_value5}\n")

            # result_file.write('#' * 50 + "\n")

            print(f"Results saved to: {result_file_path}")

def main():
    args = parse_args()
    args.window_settings = args.window_settings.split(",")
    output_path = args.data_path+'_z_score'

    if os.path.exists(output_path):
        print(f"文件已存在，跳过写入：{output_path}")
    # else:
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

        for item in tqdm(data):
            # textwm w_wm_output_attacked
            for key in [ "w_wm_output_attacked","no_wm_output","w_wm_output"]:
                input_text = item.get(key, "")  # 获取文本内容
                if input_text:  # 确保文本不为空
                    try:
                        # 调用 watermark_detector.detect 获取 score_dict
                        for window_size in args.window_settings:
                            print(window_size)
                            score_dict = watermark_detector.detect(
                                        input_text,
                                        return_prediction=False,
                                        convert_to_float=True,
                                        window_size=window_size,
                                        window_stride=1,
                                    )
                            score_dict_item = {
                                key
                                + (f"_win{window_size}-1" if window_size else "")
                                + "_"
                                + k: safe_convert(v)
                                for k, v in score_dict.items()
                            }
                            item.update(score_dict_item)

                    except ValueError as e:
                        # 捕获 ValueError 异常，并将 z_score 设置为 NaN
                        if "Must have at least 1 token" in str(e):
                            item[f"{key}_z_score"] = math.nan
                        else:
                            raise  # 如果是其他 ValueError 异常，重新抛出
        # 写回 JSONL 文件
        with open(output_path, 'w', encoding='utf-8') as file:
            for item in data:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Updated JSONL file saved to: {output_path}")

    AUROC(output_path,args=args)

if __name__ == '__main__':
    main()