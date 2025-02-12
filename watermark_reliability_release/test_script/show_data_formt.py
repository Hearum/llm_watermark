import json
import pdb
file_path = "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_5_32_0.2_LSH_v2.1_c4_news_500/gen_table_GPT.jsonl_z_score"

data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line.strip()))

print(data[0].keys())
# dict_keys(['idx', 'truncated_input', 'baseline_completion', 'orig_sample_length', 'prompt_length', 'baseline_completion_length', 'no_wm_output',
#  'w_wm_output', 'no_wm_output_length', 'w_wm_output_length', 'spike_entropies', 'w_wm_output_attacked', 'dipper_inputs_Lex20_Order0'])
print(data[0]['w_wm_output'])
for i in range(len(data)):
    print('#'*50,'w_wm_output','#'*50)
    print(data[i]['w_wm_output_z_score'])
    print(data[i]['w_wm_output'])
    print('#'*50,'w_wm_output_attacked','#'*50)
    print(data[i]['w_wm_output_attacked_z_score'])
    print(data[i]['w_wm_output_attacked'])
    print('#'*50,'no_wm_output','#'*50)
    print(data[i]['no_wm_output_z_score'])
    print(data[i]['no_wm_output'])
    # print(data[i]['dipper_inputs_Lex20_Order0'])
    pdb.set_trace()
