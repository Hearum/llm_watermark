import torch
from lsh_kwg import WatermarkLogitsProcessor  # 假设你的WatermarkLogitsProcessor类在watermark_module.py中
import sys

# 添加新的搜索路径

def main():
    # 1. 初始化 vocab 和其他参数
    vocab = [1, 2, 3, 4, 5]  # 假设只有5个token
    gamma = 0.6  # 1的比例
    delta = 1.5  # 偏置
    n_hashes = 5  # LSH函数数量
    n_features = 32  # LSH维度
    
    # 2. 实例化 WatermarkLogitsProcessor
    watermark_processor = WatermarkLogitsProcessor(
        vocab=vocab, 
        gamma=gamma, 
        delta=delta, 
        n_hashes=n_hashes, 
        n_features=n_features
    )
    
    # 3. 创建输入数据
    input_ids = torch.tensor([[1, 2, 3], [2, 3, 4]])  # 假设两个输入句子
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]])  # 假设5个token的logits分数

    # 4. 调用 __call__ 方法
    processed_scores = watermark_processor(input_ids=input_ids, scores=scores)

    # 5. 输出结果
    print("Processed Scores:")
    print(processed_scores)
    
if __name__ == "__main__":
    main()
