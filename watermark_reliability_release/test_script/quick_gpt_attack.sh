
# PATH_DIR=/home/shenhm/documents/temp/c4/KWG_TEST/len_150/llama_13B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863
# python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/gpt_attacker.py \
#     --data_path="$PATH_DIR""/gen_table.jsonl"
# bash /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/demo_gpt.sh $PATH_DIR

PATH_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/Ours_4.0_test_20/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_LSH_H_6_wikitext
python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/gpt_attacker.py \
    --data_path="$PATH_DIR""/gen_table.jsonl"
bash /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/demo_gpt.sh $PATH_DIR

PATH_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/Ours_4.0_test_20/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_16_LSH_H_16_wikitext
python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/gpt_attacker.py \
    --data_path="$PATH_DIR""/gen_table.jsonl"
bash /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/demo_gpt.sh $PATH_DIR
