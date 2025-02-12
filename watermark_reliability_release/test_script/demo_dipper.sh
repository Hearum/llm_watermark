
#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

PATH_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_self_wiki_c4_new

O=100
L=80

if [[ "$PATH_DIR" == *"KWG"* ]]; then
    python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score_kwg.py \
        --data_path="$PATH_DIR""/gen_table_dipper_O"$O"_L"$L".jsonl" \
        --config_path="$PATH_DIR""/gen_table_meta.json" 
else
    python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score.py \
        --data_path="$PATH_DIR""/gen_table_dipper_O"$O"_L"$L".jsonl" \
        --config_path="$PATH_DIR""/gen_table_meta.json" 
fi

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_auc_roc.py \
    --data_path="$PATH_DIR""/gen_table_dipper_O"$O"_L"$L".jsonl_z_score" 
