
#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

PATH_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/Ours_full/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_24_12_LSH_v3.0_c4
L=60
O=60

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/windows_script/dipper_attack.py \
    --data_path="$PATH_DIR""/gen_table.jsonl" \
    --o $O \
    --l $L \

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/windows_script/cal_z_score.py \
        --data_path="$PATH_DIR""/gen_table_dipper_O${O}_L${L}.jsonl" \
        --config_path="$PATH_DIR""/gen_table_meta.json" 

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/windows_script/cal_auc_roc.py \
    --data_path="$PATH_DIR""/gen_table_dipper_O${O}_L${L}.jsonl_z_score" 


