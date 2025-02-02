import os
import json
from copy import deepcopy

import numpy as np
import sklearn.metrics as metrics
import argparse  

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
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_eval_cp/gen_table_w_metrics.jsonl",
        help="Path to the data file containing the z-scores"
    )
    return parser.parse_args()

def load_z_scores(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    # Assuming 'w_wm_output_z_score' and 'no_wm_output_z_score' are the relevant keys
    human_z_scores = [entry['no_wm_output_z_score'] for entry in data]
    machine_z_scores = [entry['w_wm_output_attacked_z_score'] for entry in data]
    
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

def main():
    args = parse_args()

    # Load z-scores from the file
    human_z, machine_z = load_z_scores(args.data_path)
    import pdb
    pdb.set_trace()

    # Clean NaN values from z-scores
    human_z, machine_z = clean_z_scores(human_z, machine_z)

    # Calculate AUC-ROC and TPR values
    roc_auc, fpr, tpr, _ = get_roc_auc(human_z, machine_z)
    print(f"ROC AUC: {roc_auc}")

    # TPR (FPR = 0%)
    tpr_value0 = get_tpr(fpr, tpr, 0.0)
    print(f"TPR (FPR = 0%): {tpr_value0}")

    # TPR (FPR = 1%)
    tpr_value1 = get_tpr(fpr, tpr, 0.01)
    print(f"TPR (FPR = 1%): {tpr_value1}")

    # TPR (FPR = 5%)
    tpr_value5 = get_tpr(fpr, tpr, 0.05)
    print(f"TPR (FPR = 5%): {tpr_value5}")

    # Updating machine results with the calculated metrics
    machine_results = json.load(open(args.machine_fname))
    watermark_detection = deepcopy(machine_results.get('watermark_detection', {}))
    
    # Add the calculated metrics
    watermark_detection['roc_auc'] = roc_auc
    watermark_detection['TPR (FPR = 0%)'] = tpr_value0
    watermark_detection['TPR (FPR < 1%)'] = tpr_value1
    watermark_detection['TPR (FPR < 5%)'] = tpr_value5
    
    # # Save updated results back
    # machine_results['watermark_detection'] = watermark_detection
    # with open(args.machine_fname, 'w') as f:
    #     json.dump(machine_results, f, indent=4)

if __name__ == "__main__":
    main()
