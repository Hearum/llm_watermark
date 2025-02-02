"""
Adapted from the eval script in https://github.com/XiangLi1999/ContrastiveDecoding
requirement: pip install simcse (https://github.com/princeton-nlp/SimCSE)
"""
import numpy as np

from simcse import SimCSE

# 计算文本的连贯性得分（Coherence Score），衡量 生成文本（generated_text）是否与前缀文本（prefix_text）语义相关。
# 使用 SimCSE 计算相似度矩阵，并取其 对角线均值（trace mean） 作为最终连贯性得分。
def get_coherence_score(prefix_text, generated_text, 
                        model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
    
    print(len(prefix_text), len(generated_text))
    model = SimCSE(model_name)

    similarities = model.similarity(prefix_text, generated_text)
    similarities = np.array(similarities)
    coherence_score = similarities.trace() / len(similarities) 

    return coherence_score
