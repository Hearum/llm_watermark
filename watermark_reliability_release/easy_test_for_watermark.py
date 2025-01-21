import torch
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessorList
from functools import partial
from dataclasses import dataclass
from watermark_processor import WatermarkLogitsProcessor  # 假设已经正确导入


def test_watermark_logits_processor_call():
    # 1. 创建一个 WatermarkLogitsProcessor 实例
    vocab = [i for i in range(100)]  # 一个简单的词汇表，包含 100 个 token
    processor = WatermarkLogitsProcessor(
        vocab=vocab,   # 设置词表
        gamma=0.5,     # 水印的参数
        delta=2.0,     # 水印的参数
        n_hashes=5,    # LSH 哈希数量
        n_features=32, # 每个哈希的维度
        threshold_len=5  # 水印的阈值
    )

    # 2. 创建测试输入
    input_ids = torch.tensor([[1, 3, 4, 5,6,7,8,9,10,11,12], [1, 3, 4, 5,6,7,8,9,10,11,12]])  # 假设有两个序列作为输入
    scores = torch.randn(2, 100)  # 假设模型有 8 个候选词，每个候选词一个分数

    # 3. 调用 WatermarkLogitsProcessor 的 __call__ 方法
    output_scores = processor(input_ids, scores)

    # 4. 验证输出
    assert output_scores.shape == scores.shape, f"输出形状应该为 {scores.shape}，但实际为 {output_scores.shape}"
    
    # 检查输出是否已添加水印偏置
    # 由于水印的逻辑复杂，我们可以做一个简单的检查：
    # 通过打印结果，手动检查是否有水印的影响（例如偏置是否加到绿色列表的token上）

    print(scores)

    print(output_scores)

    # 还可以根据绿色token（greenlist）来验证偏置的正确性，具体测试绿色token的分数是否增加。
    # 但这个需要根据实现来推算，因为绿色token是在 `self._score_rejection_sampling()` 中计算的。

    # 测试通过的标志
    print("WatermarkLogitsProcessor__call__ 测试通过!")

# 执行测试
test_watermark_logits_processor_call()
