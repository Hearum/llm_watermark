import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载预训练的 GPT-2 模型和分词器
model_id = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"  # 自动将模型分布到多个 GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 设置模型为评估模式
model.eval()

# 输入文本
input_text = "The quick brown fox"

# 将输入文本转为 token_id
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 打印输入的 token_id 和对应的词
print("Input tokens:", input_text)
print("Input token IDs:", input_ids)

# 获取模型的输出
with torch.no_grad():
    outputs = model(input_ids)

# 输出的 logits
logits = outputs.logits

# 获取 logits 的最后两个位置
logit_minus_1 = logits[0, -1]  # logit[-1] 是下一个 token 的 logits
logit_minus_2 = logits[0, -2]  # logit[-2] 是当前生成位置前一个 token 的 logits

# 将 logits 转为概率
probabilities_minus_1 = torch.softmax(logit_minus_1, dim=-1)
probabilities_minus_2 = torch.softmax(logit_minus_2, dim=-1)

# 获取最大概率的 token
predicted_token_minus_1 = torch.argmax(probabilities_minus_1).item()
predicted_token_minus_2 = torch.argmax(probabilities_minus_2).item()

# 打印 logit[-2] 和 logit[-1] 对应的最大概率 token
print(f"Predicted token at logit[-1]: {tokenizer.decode(predicted_token_minus_1)}")
print(f"Predicted token at logit[-2]: {tokenizer.decode(predicted_token_minus_2)}")

# 打印 logit[-2] 对应的 token 的最大概率
print(f"Logit[-2] max probability token: {tokenizer.decode(predicted_token_minus_2)}")
print(f"Input token ID for the first token in input: {input_ids[0][0].item()}")
print(f"Input token corresponding to first input token: {tokenizer.decode(input_ids[0][0].item())}")
