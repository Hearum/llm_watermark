PATH_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/windows_text/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6__0.25_LSH_H_6_c4
python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/ds_attacker.py \
    --data_path="$PATH_DIR""/gen_table.jsonl"
bash /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/demo_deepseek.sh $PATH_DIR

# PATH_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width__skipgram_wikit_ff-skipgram_prf-4-True-15485863
# python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/ds_attacker.py \
#     --data_path="$PATH_DIR""/gen_table.jsonl"

# bash /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/demo_deepseek.sh $PATH_DIR

# PATH_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/c4/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width__skipgram_wikit_ff-additive_prf-4-True-15485863
# python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/ds_attacker.py \
#     --data_path="$PATH_DIR""/gen_table.jsonl"

# bash /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/demo_deepseek.sh $PATH_DIR