from transformers import AutoTokenizer

def test_tokenizer_consistency(tokenizer, input_ids):
    """
    测试给定 input_ids 通过 tokenizer 解码后再编码的结果是否一致。
    
    Args:
        tokenizer: Hugging Face 的分词器实例。
        input_ids: 待测试的输入 ID 列表。
    
    Returns:
        None
    """
    # Step 1: 解码 input_ids 到文本
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    
    # Step 2: 再次编码文本为 IDs
    re_encoded_ids = tokenizer.encode(decoded_text, add_special_tokens=False)
    
    # Step 3: 打印原始 IDs 和重新编码后的 IDs
    print("Original input_ids: ", input_ids)
    print("Decoded text: ", decoded_text)
    print("Re-encoded IDs: ", re_encoded_ids)
    
    # Step 4: 检查一致性
    if input_ids == re_encoded_ids:
        print("\n✅ Test passed: The input_ids are consistent after decoding and re-encoding.")
    else:
        print("\n❌ Test failed: The input_ids are not consistent.")
        
        # Step 5: 打印具体差异
        print("\nDifferences:")
        for i, (original_id, re_encoded_id) in enumerate(zip(input_ids, re_encoded_ids)):
            if original_id != re_encoded_id:
                print(f"Position {i}: Original ID {original_id} -> Re-encoded ID {re_encoded_id}")
        
        # 如果 re_encoded_ids 的长度不同，也需要提示
        if len(input_ids) != len(re_encoded_ids):
            print(f"\nLength mismatch: Original length {len(input_ids)} -> Re-encoded length {len(re_encoded_ids)}")

# 使用示例
if __name__ == "__main__":
    # 替换为你的模型路径
    tokenizer = AutoTokenizer.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",local_files_only=True)
    
    # 测试的 input_ids
    test_input_ids = [  822,  1243,   363,   445,   775, 26163, 20986, 27453, 24258, 28225,
         30341, 29021, 16686,  5361, 15470, 20124, 19641, 24885,  6255, 23035,
         30319, 26522, 13302,  8218,  9139, 17985, 10641, 25298,  6539, 28294,
          5743, 30745, 14468,  6684, 11705,  7930, 30387, 15739, 29243,  1945,
         13809, 16187,  6352,  9687, 28894,  7813,  6061, 12705, 17097, 11227,
         10143, 18259,  3895, 26608, 23386,  9834]
    
    # 调用测试函数
    test_tokenizer_consistency(tokenizer, test_input_ids)
