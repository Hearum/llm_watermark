################################################
# Dataset-Methord Visualization ppl
################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

colors = ['#5975A4','#CC8963','#5F9E6E','#B55D60','#857AAB','#8D7866']

# Generating synthetic data for the boxplot (since actual data isn't provided)
np.random.seed(42)

category_model = []
z_scores = []
categories = ['Watermarked', 'Watermarked-attacked', 'Un-watermarked']
models = ["WikiText-LSH","WikiText-KGW" ,"C4-LSH","C4-KGW", "LFQA-LSH", "LFQA-KGW"]


# WikiText_LSH="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/delta5_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_32_0.2_LSH_v2.2_c4_new/gen_table_GPT.jsonl_ppl"
# C4_LSH_path="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_32_0.2_LSH_v2.2_c4_new/gen_table_GPT.jsonl_ppl"
# LFQA_LSH_path="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_32_0.2_LSH_v2.2_c4_new/gen_table_GPT.jsonl_ppl"

# WikiText_KGW="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/delta5_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_selfhash_wikit/gen_table_GPT.jsonl_ppl" 
# C4_KGW_path="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_self_wiki_c4_new/gen_table_GPT.jsonl_ppl"
# LFQA_KGW_path="/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_self_wiki_c4_new/gen_table_GPT.jsonl_ppl"

import json
import numpy as np

# 路径列表
paths = {
    "WikiText_LSH": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/delta5_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_32_0.2_LSH_v2.2_c4_new/gen_table_GPT.jsonl_ppl",
    "C4_LSH": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_32_0.2_LSH_v2.2_c4_new/gen_table_GPT.jsonl_ppl",
    "LFQA_LSH": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/lfqa/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_LshParm_6_32_0.25_LSH_v2.2_lfqa_new/gen_table_GPT.jsonl_ppl",
    "WikiText_KGW": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/wikitext/delta5_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_selfhash_wikit/gen_table_GPT.jsonl_ppl",
    "C4_KGW": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/delta2_len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_width_4_self_wiki_c4_new/gen_table_GPT.jsonl_ppl",
    "LFQA_KGW": "/home/shenhm/documents/lm-watermarking/watermark_reliability_release/output/lfqa/len_150/llama_7B_N500_T200_no_filter_batch_1_delta_5_gamma_0.25_KWG_ff-anchored_minhash_prf-4-True-15485863/gen_table_GPT.jsonl_ppl"
}

# 用于存储数据的字典
all_data = {}

# 处理每个路径
for key, path in paths.items():
    data = {"w_wm_output_ppl": [], "w_wm_output_attacked_ppl": [], "no_wm_output_ppl": []}
    
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                entry = json.loads(line.strip())
                
                if 'w_wm_output_ppl' in entry:
                    data["w_wm_output_ppl"].append(entry["w_wm_output_ppl"])
                if 'w_wm_output_attacked_ppl' in entry:
                    data["w_wm_output_attacked_ppl"].append(entry["w_wm_output_attacked_ppl"])
                if 'no_wm_output_ppl' in entry:
                    data["no_wm_output_ppl"].append(entry["no_wm_output_ppl"])
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}, 跳过这一行")
    
    all_data[key] = data

# 模拟 Z-Score 数据
data = {
    "Watermarked": {
        "WikiText-LSH": np.array(all_data["WikiText_LSH"]['w_wm_output_ppl']),
        "WikiText-KGW": np.array(all_data["WikiText_KGW"]['w_wm_output_ppl']),
        "C4-LSH": np.array(all_data["C4_LSH"]['w_wm_output_ppl']),
        "C4-KGW": np.array(all_data["C4_KGW"]['w_wm_output_ppl']),
        "LFQA-LSH": np.array(all_data["LFQA_LSH"]['w_wm_output_ppl']),
        "LFQA-KGW": np.array(all_data["LFQA_KGW"]['w_wm_output_ppl']),
    },
    "Watermarked-attacked": {
        "WikiText-LSH": np.array(all_data["WikiText_LSH"]['w_wm_output_attacked_ppl']),
        "WikiText-KGW": np.array(all_data["WikiText_KGW"]['w_wm_output_attacked_ppl']),
        "C4-LSH": np.array(all_data["C4_LSH"]['w_wm_output_attacked_ppl']),
        "C4-KGW": np.array(all_data["C4_KGW"]['w_wm_output_attacked_ppl']),
        "LFQA-LSH": np.array(all_data["LFQA_LSH"]['w_wm_output_attacked_ppl']),
        "LFQA-KGW": np.array(all_data["LFQA_KGW"]['w_wm_output_attacked_ppl']),
    },
    "Un-watermarked": {
        "WikiText-LSH": np.array(all_data["WikiText_LSH"]['no_wm_output_ppl']),
        "WikiText-KGW": np.array(all_data["WikiText_KGW"]['no_wm_output_ppl']),
        "C4-LSH": np.array(all_data["C4_LSH"]['no_wm_output_ppl']),
        "C4-KGW": np.array(all_data["C4_KGW"]['no_wm_output_ppl']),
        "LFQA-LSH": np.array(all_data["LFQA_LSH"]['no_wm_output_ppl']),
        "LFQA-KGW": np.array(all_data["LFQA_KGW"]['no_wm_output_ppl']),
    }
}

category_list = []
model_list = []
zscore_list = []

for category in categories:
    for model in models:
        # Check that the z-scores have the same length
        zscores = data[category][model][:200]
        # Extend the lists based on the actual data length
        category_list.extend([category] * len(zscores))
        model_list.extend([model] * len(zscores))
        zscore_list.extend(zscores)

df_corrected = pd.DataFrame({
    "Category": category_list,
    "Model": model_list,
    "Z-Score": zscore_list
})

# 设置绘图
plt.figure(figsize=(10, 8),dpi=300)

# 创建箱型图
import seaborn as sns
sns.set(style="whitegrid")

# 定制离群点的样式为菱形
flierprops = dict(marker='D', markersize=6, alpha=0.7)

ax = sns.boxplot(
    x="Category", y="Z-Score", hue="Model", data=df_corrected,
    palette=colors, showfliers=True, linewidth=2.0)

plt.legend(title="Dataset-Model", loc="upper right", bbox_to_anchor=(1, 0))
# 定制化图表
ax.set_ylabel('Z-Score', fontsize=14)  # 增加字体大小
ax.set_xlabel('')
ax.set_xticklabels(['Watermarked', 'Watermarked-attacked', 'Un-watermarked'], fontsize=14)
# ax.set_title("Boxplot of Z-Scores by Dataset Type and Model")
plt.legend(title="Dataset-Method", bbox_to_anchor=(0.78, 0.95), loc="upper left")

plt.tight_layout()
plt.show()
