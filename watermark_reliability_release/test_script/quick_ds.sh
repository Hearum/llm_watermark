
PATH_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_32_0.2_LSH_v2.2_c4_new
python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/ds_attacker.py \
    --data_path="$PATH_DIR""/gen_table.jsonl"

bash /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/demo_deepseek.sh $PATH_DIR


# PATH_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_self_wiki_c4_new
# python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/ds_attacker.py \
#     --data_path="$PATH_DIR""/gen_table.jsonl"
# bash /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/demo_deepseek.sh $PATH_DIR
