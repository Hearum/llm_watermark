
#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

PATH_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/ours3_0/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_0.3_LSH_v3.0_c4
O=60
L=60

# python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/dipper_attack.py \
#     --data_path="$PATH_DIR""/gen_table.jsonl" \
#     --o $O \
#     --l $L \

if [[ "$PATH_DIR" == *"KWG"* ]]; then
    python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score_kwg.py \
        --data_path="$PATH_DIR""/gen_table_dipper_O${O}_L${L}.jsonl" \
        --config_path="$PATH_DIR""/gen_table_meta.json" 
else
    python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score.py \
        --data_path="$PATH_DIR""/gen_table_dipper_O${O}_L${L}.jsonl" \
        --config_path="$PATH_DIR""/gen_table_meta.json" 
fi

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_auc_roc.py \
    --data_path="$PATH_DIR""/gen_table_dipper_O${O}_L${L}.jsonl_z_score" 

# L=20
# O=20

# python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/dipper_attack.py \
#     --data_path="$PATH_DIR""/gen_table.jsonl" \
#     --o $O \
#     --l $L \

# if [[ "$PATH_DIR" == *"KWG"* ]]; then
#     python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score_kwg.py \
#         --data_path="$PATH_DIR""/gen_table_dipper_O${O}_L${L}.jsonl" \
#         --config_path="$PATH_DIR""/gen_table_meta.json" 
# else
#     python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score.py \
#         --data_path="$PATH_DIR""/gen_table_dipper_O${O}_L${L}.jsonl" \
#         --config_path="$PATH_DIR""/gen_table_meta.json" 
# fi

# python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_auc_roc.py \
#     --data_path="$PATH_DIR""/gen_table_dipper_O${O}_L${L}.jsonl_z_score" 

# L=20
# O=20

# python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/dipper_attack.py \
#     --data_path="$PATH_DIR""/gen_table.jsonl" \
#     --o $O \
#     --l $L \

# if [[ "$PATH_DIR" == *"KWG"* ]]; then
#     python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score_kwg.py \
#         --data_path="$PATH_DIR""/gen_table_dipper_O${O}_L${L}.jsonl" \
#         --config_path="$PATH_DIR""/gen_table_meta.json" 
# else
#     python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score.py \
#         --data_path="$PATH_DIR""/gen_table_dipper_O${O}_L${L}.jsonl" \
#         --config_path="$PATH_DIR""/gen_table_meta.json" 
# fi

# python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_auc_roc.py \
#     --data_path="$PATH_DIR""/gen_table_dipper_O${O}_L${L}.jsonl_z_score" 
