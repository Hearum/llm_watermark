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
        "--thread_base",
        type=float,
        default=4,
        help="File name of human code detection results",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/spoofing_attack/formot_signle_kwg_h_6_spoofing_dolly-writing-100-long_8.5.jsonl",
        help="Path to the data file containing the z-scores"
    )
    return parser.parse_args()

import pdb
import numpy as np

def main():
    args = parse_args()
    print(args.data_path)

    data = []
    with open(args.data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    all_len = len(data)
    # dict_keys(['idx', 'prompt', 'w_wm_output_attacked', 'w_wm_output_attacked_z_socre', 'w_wm_output', 'w_wm_output_z_score'])
    spoof_z_score = np.array([float(item['w_wm_output_attacked_z_socre']) for item in data if item['w_wm_output_attacked_z_socre'] is not None])
    print(np.sum(spoof_z_score>args.thread_base)/all_len)
    
    

    result_file_path = os.path.splitext(args.data_path)[0] + f"_aucroc_result.txt"
    # Open the result file for writing
    with open(result_file_path, 'w') as result_file:
        # Write the header for the results
        result_file.write(f"Results for {args.data_path}\n")
        result_file.write("=" * 50 + "\n")
        result_file.write("resulte:" +f"{np.sum(spoof_z_score>args.thread_base)/all_len}" + "\n")
    print(f"Results saved to: {result_file_path}")



if __name__ == "__main__":
    main()
