#!/bin/bash

cd /home/shenhm/documents/lm-watermarking/watermark_reliability_release
export HF_HOME=/home/shenhm/doucuments/lm-watermarking/watermark_reliability_release/dataset
export HF_ENDPOINT=https://hf-mirror.com


export CUDA_VISIBLE_DEVICES=3

MODEL_NAME=llama_7B
DATASET=c4
#    --dataset_name=wikitext \
#    --dataset_config_name=wikitext-103-raw-v1 \
#     --dataset_name=c4
# openai_humaneval
# codeparrot
DELTA=5
GAMMA=0.25

N_HASHES=6
THRESHOLD=0.25
GENERATE_LEN=150
H=6

# LLAMA 13b /home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1
# LLAMA 7b /home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9

if [[ "$MODEL_NAME" == "llama_13B" ]]; then
    MODEL_PATH='/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1'
elif [[ "$MODEL_NAME" == "llama_7B" ]]; then
    MODEL_PATH='/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9'
elif [[ "$MODEL_NAME" == "opt_6.7B" ]]; then
    MODEL_PATH='/home/shenhm/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0'
elif [[ "$MODEL_NAME" == "opt_1.3B" ]]; then
    MODEL_PATH='/home/shenhm/.cache/huggingface/hub/models--facebook--opt-1.3b/snapshots/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62'
elif [[ "$MODEL_NAME" == "gpt2" ]]; then
    MODEL_PATH='/home/shenhm/.cache/huggingface/hub/models--gpt2-xl/snapshots/15ea56dee5df4983c59b2538573817e1667135e2'
elif [[ "$MODEL_NAME" == "Mistral_7B" ]]; then
    MODEL_PATH='/home/shenhm/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/2dcff66eac0c01dc50e4c41eea959968232187fe'
else
    MODEL_PATH='/path/to/default/model'
fi


OUTPUT_DIR=/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/${DATASET}/windows_new/len_${GENERATE_LEN}
RUN_NAME=${MODEL_NAME}_N500_T200_no_filter_batch_1_delta_${DELTA}_gamma_${GAMMA}_LshParm_${N_HASHES}_${N_FEATURES}_${THRESHOLD}_LSH_H_${H}_${DATASET}

GENERATION_OUTPUT_DIR="$OUTPUT_DIR"/"$RUN_NAME"
echo "Running generation pipeline with output dir: $GENERATION_OUTPUT_DIR"

python generation_pipeline.py \
    --model_name=$LLAMA_PATH \
    --dataset_name=$DATASET \
    --dataset_config_name=wikitext-103-raw-v1  \
    --max_new_tokens=150 \
    --model_max_generation_tokens=200 \
    --min_generations=500 \
    --input_truncation_strategy=prompt_length \
    --min_prompt_tokens=100 \
    --input_filtering_strategy=prompt_and_completion_length \
    --output_filtering_strategy=max_new_tokens \
    --seeding_scheme=selfhash \
    --gamma=$GAMMA \
    --delta=$DELTA \
    --n_hashes=$N_HASHES \
    --threshold=$THRESHOLD \
    --run_name="$RUN_NAME"_gen \
    --wandb=True \
    --verbose=True \
    --output_dir=$GENERATION_OUTPUT_DIR \
    --model_name_or_path $MODEL_PATH  \
    --generation_batch_size 1 \
    --LSH=False \
    --LSH_windows=True \
    --h_win=$H  \


O=60
L=60

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/dipper_attack.py \
    --data_path="$GENERATION_OUTPUT_DIR""/gen_table.jsonl" \
    --o $O \
    --l $L \

if [[ "$GENERATION_OUTPUT_DIR" == *"KWG"* ]]; then
    python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score_kwg.py \
        --data_path="$GENERATION_OUTPUT_DIR""/gen_table_dipper_O"$O"_L"$L".jsonl" \
        --config_path="$GENERATION_OUTPUT_DIR""/gen_table_meta.json" 
else
    python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score.py \
        --data_path="$GENERATION_OUTPUT_DIR""/gen_table_dipper_O"$O"_L"$L".jsonl" \
        --config_path="$GENERATION_OUTPUT_DIR""/gen_table_meta.json" 
fi

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_auc_roc.py \
    --data_path="$GENERATION_OUTPUT_DIR""/gen_table_dipper_O${O}_L${L}.jsonl_z_score" 


O=20
L=20

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/dipper_attack.py \
    --data_path="$GENERATION_OUTPUT_DIR""/gen_table.jsonl" \
    --o $O \
    --l $L \

if [[ "$GENERATION_OUTPUT_DIR" == *"KWG"* ]]; then
    python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score_kwg.py \
        --data_path="$GENERATION_OUTPUT_DIR""/gen_table_dipper_O"$O"_L"$L".jsonl" \
        --config_path="$GENERATION_OUTPUT_DIR""/gen_table_meta.json" 
else
    python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_z_score.py \
        --data_path="$GENERATION_OUTPUT_DIR""/gen_table_dipper_O"$O"_L"$L".jsonl" \
        --config_path="$GENERATION_OUTPUT_DIR""/gen_table_meta.json" 
fi

python /home/shenhm/documents/lm-watermarking/watermark_reliability_release/test_script/cal_auc_roc.py \
    --data_path="$GENERATION_OUTPUT_DIR""/gen_table_dipper_O${O}_L${L}.jsonl_z_score" 