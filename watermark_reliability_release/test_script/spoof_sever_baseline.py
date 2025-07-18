import os
import json
from copy import deepcopy

import numpy as np
import sklearn.metrics as metrics
import argparse  
import json
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--human_fname",
        type=str,
        default="outputs_human",
        help="File name of human code detection results",
    )
    parser.add_argument(
        "--machine_fname",
        type=str,
        default="outputs",
        help="File name of machine code detection results",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/KWG_TEST/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-3-True-15485863/gen_table_dipper_O60_L60.jsonl_z_score",
        help="Path to the data file containing the z-scores"
    )
    return parser.parse_args()

def load_z_scores(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    # Assuming 'w_wm_output_z_score' and 'no_wm_output_z_score' are the relevant keys
    human_z_scores = [entry.get('no_wm_output_z_score', math.nan) for entry in data]
    # w_wm_output_attacked_z_score  w_wm_output_z_score
    machine_z_scores = [entry.get('w_wm_output_attacked_z_score', math.nan) for entry in data]
    return human_z_scores, machine_z_scores

def load_z_scores_2(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    # Assuming 'w_wm_output_z_score' and 'no_wm_output_z_score' are the relevant keys
    human_z_scores = [entry.get('no_wm_output_z_score', math.nan) for entry in data]
    # w_wm_output_attacked_z_score  w_wm_output_z_score
    machine_z_scores = [entry.get('w_wm_output_z_score', math.nan) for entry in data]
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

import os
import pdb

def main():
    args = parse_args()
    print(args.data_path)
    
    # 结果保存文件路径
    
    result_file_path = os.path.splitext(args.data_path)[0] + f"sever_baseline.txt"

    # Open the result file for writing
    with open(result_file_path, 'w') as result_file:
        # Write the header for the results
        result_file.write(f"Results for {args.data_path}\n")
        result_file.write("=" * 50 + "\n")

        # First run
        print("Processing first z-scores...")
        human_z, machine_z = load_z_scores_2(args.data_path)
        machine_z=machine_z+machine_z
        human_z= human_z+human_z
        pdb.set_trace()
        # Clean NaN values from z-scores
        human_z, machine_z = clean_z_scores(human_z, machine_z)

        # Calculate AUC-ROC and TPR values
        roc_auc, fpr, tpr, thresholds = get_roc_auc(human_z, machine_z)

        print(f"ROC AUC: {roc_auc}")
        result_file.write(f"ROC AUC: {roc_auc}\n")

        # 找到最接近 FPR = 0.1% (0.001) 的索引
        idx = (np.abs(fpr - 0.001)).argmin()

        
        z_score_at_fpr_001 = thresholds[idx]
        print("z_score_at_fpr_001",z_score_at_fpr_001)
   
        result_file.write(f"z_score_at_fpr_0001: {z_score_at_fpr_001}\n")

        # TPR (FPR = 0.1%)
        tpr_value1 = get_tpr(fpr, tpr, 0.001)

        print(f"TPR (FPR = 0.1%): {tpr_value1}")
        result_file.write(f"TPR (FPR = 0.1%): {tpr_value1}\n")

        result_file.write('#' * 50 + "\n")
        pdb.set_trace()

    print(f"Results saved to: {result_file_path}")


    # # Updating machine results with the calculated metrics
    # machine_results = json.load(open(args.machine_fname))
    # watermark_detection = deepcopy(machine_results.get('watermark_detection', {}))
    
    # # Add the calculated metrics
    # watermark_detection['roc_auc'] = roc_auc
    # watermark_detection['TPR (FPR = 0%)'] = tpr_value0
    # watermark_detection['TPR (FPR < 1%)'] = tpr_value1
    # watermark_detection['TPR (FPR < 5%)'] = tpr_value5
    
    # # Save updated results back
    # machine_results['watermark_detection'] = watermark_detection
    # with open(args.machine_fname, 'w') as f:
    #     json.dump(machine_results, f, indent=4)

if __name__ == "__main__":
    main()
