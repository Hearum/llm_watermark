#!/bin/bash

cd /home/shenhm/documents/lm-watermarking/watermark_reliability_release
export HF_HOME=/home/shenhm/doucuments/lm-watermarking/watermark_reliability_release/dataset
export HF_ENDPOINT=https://hf-mirror.com

OUTPUT_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/delta5_len_150

RUN_NAME=llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_addhash_wikit

GENERATION_OUTPUT_DIR="$OUTPUT_DIR"/"$RUN_NAME"

echo "Running generation pipeline with output dir: $GENERATION_OUTPUT_DIR"

#    --dataset_name=wikitext \
#    --dataset_config_name=wikitext-103-raw-v1 \
#     --dataset_name=c4
# codeparrot
# openai_humaneval

export CUDA_VISIBLE_DEVICES=2
# ff-anchored_minhash_prf-4-True-15485863
# skipgram
# ff-additive_prf-4-True-15485863
python generation_pipeline.py \
    --model_name=$LLAMA_PATH \
    --dataset_name=wikitext \
    --dataset_config_name=wikitext-103-raw-v1 \
    --max_new_tokens=150 \
    --model_max_generation_tokens=150 \
    --min_generations=500 \
    --input_truncation_strategy=prompt_length  \
    --min_prompt_tokens=200 \
    --input_filtering_strategy=prompt_and_completion_length \
    --output_filtering_strategy=max_new_tokens \
    --seeding_scheme=ff-additive_prf-4-True-15485863 \
    --run_name="$RUN_NAME"_gen \
    --gamma=0.25 \
    --delta=5 \
    --wandb=True \
    --verbose=True \
    --output_dir=$GENERATION_OUTPUT_DIR \
    --model_name_or_path "/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9" \
    --generation_batch_size 1 \
    --LSH=False 

O=60
L=60

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/dipper_attack.py \
    --data_path="$GENERATION_OUTPUT_DIR""/gen_table.jsonl" \
    --o $O \
    --l $L \

if [[ "$PATH_DIR" == *"KWG"* ]]; then
    python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score_kwg.py \
        --data_path="$PATH_DIR/gen_table_dipper_O${O}_L${L}.jsonl"\
        --config_path="$PATH_DIR""/gen_table_meta.json" 
else
    python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score.py \
        --data_path="$PATH_DIR/gen_table_dipper_O${O}_L${L}.jsonl"\
        --config_path="$PATH_DIR/gen_table_meta.json" 
fi

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_auc_roc.py \
    --data_path="$PATH_DIR""/gen_table_dipper_O${O}_L${L}.jsonl_z_score" 


O=20
L=20

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/dipper_attack.py \
    --data_path="$GENERATION_OUTPUT_DIR""/gen_table.jsonl" \
    --o $O \
    --l $L \

if [[ "$PATH_DIR" == *"KWG"* ]]; then
    python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score_kwg.py \
        --data_path="$PATH_DIR/gen_table_dipper_O${O}_L${L}.jsonl"\
        --config_path="$PATH_DIR""/gen_table_meta.json" 
else
    python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score.py \
        --data_path="$PATH_DIR/gen_table_dipper_O${O}_L${L}.jsonl"\
        --config_path="$PATH_DIR/gen_table_meta.json" 
fi

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_auc_roc.py \
    --data_path="$PATH_DIR""/gen_table_dipper_O${O}_L${L}.jsonl_z_score" 
