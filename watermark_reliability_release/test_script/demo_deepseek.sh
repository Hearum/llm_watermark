#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

if [ -z "$1" ]; then
    echo "没有提供 PATH_DIR,使用默认路径。"
    PATH_DIR="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_32_0.2_LSH_v2.2_c4_new"
else
    PATH_DIR="$1"
    echo "使用提供的 PATH_DIR: $PATH_DIR"
fi

if [[ "$PATH_DIR" == *"KWG"* ]]; then
    python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score_kwg.py \
        --data_path="$PATH_DIR""/gen_table_deepseek_attacker.jsonl" \
        --config_path="$PATH_DIR""/gen_table_meta.json" 
        #--seeding_scheme=ff-anchored_minhash_prf-4-True-15485863\
else
    python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score_windows.py \
        --data_path="$PATH_DIR""/gen_table_deepseek_attacker.jsonl" \
        --config_path="$PATH_DIR""/gen_table_meta.json" 
fi

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_auc_roc.py \
    --data_path="$PATH_DIR""/gen_table_deepseek_attacker.jsonl_z_score" 

