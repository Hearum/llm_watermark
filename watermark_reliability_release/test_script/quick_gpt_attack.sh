

PATH_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/delta5_len_200/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_new_minhash_wiki_wikitext_new
python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/gpt_attacker.py \
    --data_path="$PATH_DIR""/gen_table.jsonl"

PATH_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/delta5_len_200/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_5_32_0.2_LSH_v2.2_wikitext_new
python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/gpt_attacker.py \
    --data_path="$PATH_DIR""/gen_table.jsonl"



