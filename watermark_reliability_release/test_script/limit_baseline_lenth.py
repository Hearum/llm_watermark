import json
import nltk
from nltk.tokenize import word_tokenize

# 下载 nltk 分词器（如未安装）
nltk.download("punkt")

# 读取 JSONL 数据
input_file = "data.jsonl"  # 原始 JSONL 文件
output_file = "data_modified.jsonl"  # 处理后 JSONL 文件

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        if not line.strip():  # 跳过空行
            continue
        
        data = json.loads(line)  # 解析 JSON 行
        
        # 处理 baseline_completion
        if "baseline_completion" in data and isinstance(data["baseline_completion"], str):
            words = word_tokenize(data["baseline_completion"])  # 分词
            truncated_text = " ".join(words[:140])  # 只保留前 140 个单词
            data["baseline_completion"] = truncated_text  # 更新文本
            data["baseline_completion_length"] = 140  # 更新长度信息
        
        # 写入新 JSONL 文件
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write("\n")  # 每行一个 JSON
