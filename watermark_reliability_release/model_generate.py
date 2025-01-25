import torch

def model_generate(model, tokenizer, input_ids, max_new_tokens=50, do_sample=False, top_k=None, top_p=None, temperature=1.0, logits_processor=None):
    """
    :param model: 预训练模型
    :param tokenizer: 用于编码和解码的tokenizer
    :param input_ids: 输入的 ID（通常是经过编码的文本）
    :param max_length: 最大生成长度
    :param do_sample: 是否使用采样（True 为采样，False 为贪婪搜索）
    :param top_k: 在采样时，保留的 top-k 个概率最高的候选
    :param top_p: 在采样时，保留的累计概率 p 以内的候选
    :param temperature: 生成时的温度控制
    :return: 生成的文本
    """
    # 初始化生成的 ID（包含输入 ID）
    generated_ids = input_ids

    # 用模型生成直到达到 max_length
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # 获取模型的输出
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]  # 获取最后一个 token 的 logits

            if logits_processor is not None:
                for processor in logits_processor:
                    logits = processor(logits)

            # 进行采样或贪婪搜索
            if do_sample:
                # 使用温度控制 logits
                logits = logits / temperature

                # 如果指定了 top_k，限制 logits 到 top_k 个候选
                if top_k is not None:
                    top_k_indices = logits.topk(top_k, dim=-1).indices
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, 0)

                # 如果指定了 top_p，限制 logits 到累计概率 p 以内的候选
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_keep = cumulative_probs <= top_p
                    sorted_indices_to_keep = sorted_indices_to_keep.cumsum(dim=-1) == 1
                    indices_to_keep = sorted_indices.gather(-1, sorted_indices_to_keep.long())
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, indices_to_keep, 0)
                    
                # 采样一个 token
                next_token_id = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), 1)
            else:
                # 贪婪搜索：选择最大概率的 token
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            # 将生成的 token 添加到生成的 ID 中
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            # 如果生成的是终止 token（如 eos_token），则停止生成
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return generated_ids