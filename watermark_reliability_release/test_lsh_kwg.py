import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessorList
from functools import partial
from dataclasses import dataclass
from lsh_kwg import WatermarkLogitsProcessor  # 假设已经正确导入

# 定义参数类用于配置
@dataclass
class Args:
    gamma: float = 0.25
    delta: float = 1.5
    seeding_scheme: str = 'default'  # 这里可以根据实际需求修改
    select_green_tokens: bool = False
    max_new_tokens: int = 50
    use_sampling: bool = True
    sampling_temp: float = 1.0
    n_beams: int = 5
    prompt_max_length: int = 128
    generation_seed: int = 42
    seed_separately: bool = False
    is_decoder_only_model: bool = True


def generate(prompt, args, model, device, tokenizer):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """
    
    print(f"Generating with {args}")
    
    # 创建 WatermarkLogitsProcessor
    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    select_green_tokens=args.select_green_tokens)
    
    # 生成参数设置
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))

    # 创建偏函数来生成文本
    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )

    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]),
        **gen_kwargs
    )

    # 处理提示文本的最大长度
    if args.prompt_max_length:
        pass
    elif hasattr(model.config, "max_position_embeddings"):
        args.prompt_max_length = model.config.max_position_embeddings - args.max_new_tokens
    else:
        args.prompt_max_length = 2048 - args.max_new_tokens

    # Tokenizer准备输入
    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True,
                           max_length=args.prompt_max_length).to(device)
    
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    # 设置随机种子并生成不带水印的文本
    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)

    # 设置种子并生成带水印的文本
    if args.seed_separately:
        torch.manual_seed(args.generation_seed)
    output_with_watermark = generate_with_watermark(**tokd_input)

    # 如果是decoder-only模型，提取新生成的token
    if args.is_decoder_only_model:
        output_without_watermark = output_without_watermark[:, tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:, tokd_input["input_ids"].shape[-1]:]

    # 解码输出结果
    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark,
            decoded_output_with_watermark,
            args)


# 测试用例
def test_watermark_generation():

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",local_files_only=True,device_map = "auto")
    device =model.device

    # 配置参数
    args = Args(gamma=0.3, delta=1.2, max_new_tokens=100, use_sampling=True, sampling_temp=0.9, n_beams=3)

    # 设置测试文本
    prompt = "Once upon a time, in a land far away,"

    # 生成水印文本
    redecoded_input, truncation_warning, output_without_watermark, output_with_watermark, _ = generate(
        prompt, args, model, device, tokenizer)

    # 打印结果
    print(f"Prompt: {redecoded_input}")
    print(f"Truncation warning: {truncation_warning}")
    print(f"Generated text without watermark: {output_without_watermark}")
    print(f"Generated text with watermark: {output_with_watermark}")


if __name__ == "__main__":
    test_watermark_generation()

# import torch
# import time
# from lsh_kwg import WatermarkLogitsProcessor  # 假设你的WatermarkLogitsProcessor类在watermark_module.py中

# def main():
#     # 1. 初始化 vocab 和其他参数
#     vocab = list(range(65400))  # 假设词汇表有100个token
#     gamma = 0.25  # 水印比例
#     delta = 1.5  # 偏置
#     n_hashes = 10  # LSH函数数量
#     n_features = 32  # LSH维度
    
#     # 2. 实例化 WatermarkLogitsProcessor
#     watermark_processor = WatermarkLogitsProcessor(
#         vocab=vocab, 
#         gamma=gamma, 
#         delta=delta, 
#         n_hashes=n_hashes, 
#         n_features=n_features
#     )

#     # 3. 准备测试数据
#     input_ids = torch.tensor([[i for i in range(10)]])  # 假设序列长度为10
#     scores = torch.randn((1, 65400))  # 假设为1个batch，10个token的logits分数，随机生成
    
#     # 4. 测试处理时间
#     start_time = time.time()  # 记录开始时间
    
#     # 5. 调用 WatermarkLogitsProcessor 的 __call__ 方法处理数据
#     processed_scores = watermark_processor(input_ids=input_ids, scores=scores)
    
#     # 6. 计算处理时间
#     end_time = time.time()  # 记录结束时间
#     processing_time = end_time - start_time  # 计算时间差
    
#     # 7. 打印输出
#     print(f"Processed Scores: {processed_scores}")
#     print(f"Time taken for processing: {processing_time:.6f} seconds")  # 打印耗时

# if __name__ == "__main__":
#     main()
