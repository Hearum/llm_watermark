from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
)
# 加载模型和分词器
model_name = "gpt2-xl"
model = AutoModelForCausalLM.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/2dcff66eac0c01dc50e4c41eea959968232187fe",local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/2dcff66eac0c01dc50e4c41eea959968232187fe",local_files_only=True)

input_text = "hello,introduce yourself"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(inputs['input_ids'], max_length=50)
result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result_text)