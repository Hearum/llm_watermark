# Script to run the generation, attack, and evaluation steps of the pipeline

# requires some OUTPUT_DIR to be set in the environment
# as well as a path to the hf format LLAMA model
cd /home/shenhm/documents/lm-watermarking/watermark_reliability_release
export HF_HOME=/home/shenhm/doucuments/lm-watermarking/watermark_reliability_release/dataset
export HF_ENDPOINT=https://hf-mirror.com

OUTPUT_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/debug

RUN_NAME=llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_new_minhash_wiki_c4_new
GENERATION_OUTPUT_DIR="$OUTPUT_DIR"/"$RUN_NAME"

echo "Running generation pipeline with output dir: $GENERATION_OUTPUT_DIR"

#    --dataset_name=wikitext \
#    --dataset_config_name=wikitext-103-raw-v1 \
#     --dataset_name=c4
# codeparrot

export CUDA_VISIBLE_DEVICES=0
# ff-anchored_minhash_prf-4-True-15485863
# skipgram_prf
# ff-additive_prf-4-True-15485863
python generation_pipeline.py \
    --model_name=$LLAMA_PATH \
    --dataset_name=c4\
    --dataset_config_name=wikitext-103-raw-v1 \
    --max_new_tokens=100 \
    --min_generations=500 \
    --input_truncation_strategy=prompt_length \
    --min_prompt_tokens=100 \
    --input_filtering_strategy=prompt_and_completion_length \
    --output_filtering_strategy=max_new_tokens \
    --seeding_scheme=ff-anchored_minhash_prf-4-True-15485863 \
    --gamma=0.25 \
    --delta=5 \
    --n_hashes=5 \
    --n_features=32 \
    --threshold=0.2 \
    --run_name="$RUN_NAME"_gen \
    --wandb=True \
    --verbose=True \
    --output_dir=$GENERATION_OUTPUT_DIR \
    --model_name_or_path "/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9" \
    --generation_batch_size 1 \
    --LSH=False \

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/test_script/dipper_attack.py \
    --data_path="$GENERATION_OUTPUT_DIR""/gen_table.jsonl" \

# --attack_method=gpt \
# python attack_pipeline.py \
#     --attack_method=dipper \
#     --run_name="$RUN_NAME"_dipper_attack \
#     --wandb=True \
#     --input_dir=$GENERATION_OUTPUT_DIR \
#     --verbose=True \
#     --order 40 \
#     --lex 40 \
#     --overwrite_output_file=True \

# python evaluation_pipeline.py \
#     --evaluation_metrics=all \
#     --run_name="$RUN_NAME"_eval \
#     --wandb=True \
#     --input_dir=$GENERATION_OUTPUT_DIR \
#     --output_dir="$GENERATION_OUTPUT_DIR"_eval \
#     --roc_test_stat=all