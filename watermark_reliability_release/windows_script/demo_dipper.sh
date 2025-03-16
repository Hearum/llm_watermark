
#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

PATH_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/windows/c4/h_16/0.jsonl
L=60
O=60

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/windows_script/dipper_attack.py \
    --data_path=$PATH_DIR \
    --o $O \
    --l $L \


# python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/windows_script/cal_z_score.py \
#     --data_path=$PATH_DIR

# python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/windows_script/cal_auc_roc.py \
#     --data_path="$PATH_DIR""/gen_table_dipper_O${O}_L${L}.jsonl_z_score" 
