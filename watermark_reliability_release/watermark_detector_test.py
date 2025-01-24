import unittest
import torch
from transformers import AutoTokenizer
from watermark_processor import *
import torch
from transformers import PreTrainedTokenizerFast



# 初始化设备和 tokenizer
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",local_files_only=True)

# 初始化 WatermarkDetector 实例
detector = WatermarkDetector(
    device=device,
    tokenizer=tokenizer,
    vocab=list(tokenizer.get_vocab().values()),
    z_threshold=4.0,
    normalizers=["unicode"],
    ignore_repeated_ngrams=False,
)

# 测试文本
test_text = """
Once upon a time, in a land far away, there lived a beautiful princess. She was the most beautiful princess in all the land. She had long, flowing, golden hair and sparkling blue eyes. Her skin was as white as snow and her lips were as red as roses. She was the envy of all the other princesses in the land.
One day, the princess was walking through the woods when she came upon a small cottage. The cottage was made of wood and had a
"""
w_wm_output="""Once upon a time, in a land far away, there was a little boy. The little boy was very happy. The little boy was very healthy. The little boy was very happy. The little boy was very healthy. The little boy was very happy. The little boy was very healthy. The little boy was very happy. The little boy was very healthy. The little boy was very happy. The little boy was very healthy. The little boy was very happy. The little boy was very healthy. The little boy was
"""

# 调用 detect 方法进行检测
result = detector.detect(text=test_text)

# 输出结果
print("raw_test_text检测结果:")
for key, value in result.items():
    print(f"{key}: {value}")

result = detector.detect(text=w_wm_output)
print("w_wm_output检测结果:")
for key, value in result.items():
    print(f"{key}: {value}")
# class TestWatermarkDetector(unittest.TestCase):

#     def setUp(self):
#         """Set up the test environment and initialize the WatermarkDetector."""

#         self.tokenizer = AutoTokenizer.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",local_files_only=True)
#         #self.model = AutoModelForCausalLM.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",local_files_only=True,device_map = "auto")
#         #self.device =model.device

#         self.device = torch.device("cpu")
#         #self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#         self.z_threshold = 4.0
#         self.normalizers = ["unicode"]
#         self.detector = WatermarkDetector(
#             vocab=list(self.tokenizer.get_vocab().values()),
#             device=self.device,
#             tokenizer=self.tokenizer,
#             z_threshold=self.z_threshold,
#             normalizers=self.normalizers,
#         )

#     def test_detect_raw_text(self):
#         """Test the detect method with raw text."""
#         text = "This is a test sentence."
#         result = self.detector.detect(text=text, return_prediction=True, return_scores=True)
#         self.assertIn("prediction", result)
#         self.assertIn("z_score", result)
#         self.assertIsInstance(result["prediction"], bool)
#         self.assertIsInstance(result["z_score"], float)

#     def test_detect_tokenized_text(self):
#         """Test the detect method with tokenized text."""
#         text = "Another test sentence for watermarks."
#         tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
#         result = self.detector.detect(tokenized_text=tokenized_text, return_prediction=True, return_scores=True)
#         self.assertIn("prediction", result)
#         self.assertIn("z_score", result)
#         self.assertIsInstance(result["prediction"], bool)
#         self.assertIsInstance(result["z_score"], float)

#     def test_detect_with_custom_z_threshold(self):
#         """Test the detect method with a custom z_threshold."""
#         text = "Testing with a custom z_threshold."
#         custom_threshold = 5.0
#         result = self.detector.detect(text=text, z_threshold=custom_threshold, return_prediction=True, return_scores=True)
#         self.assertIn("prediction", result)
#         self.assertIn("z_score", result)
#         self.assertIsInstance(result["prediction"], bool)
#         self.assertTrue(result["z_score"] <= custom_threshold or not result["prediction"])

#     # def test_detect_with_normalization(self):
#     #     """Test the detect method with text normalization."""
#     #     text = "Ｔｈｉｓ　ｉｓ　ａ　ｔｅｓｔ　ｗｉｔｈ　ｎｏｒｍａｌｉｚｅｄ　ｔｅｘｔ."
#     #     result = self.detector.detect(text=text, return_prediction=True, return_scores=True)
#     #     self.assertIn("prediction", result)
#     #     self.assertIn("z_score", result)
#     #     self.assertIsInstance(result["prediction"], bool)
#     #     self.assertIsInstance(result["z_score"], float)

#     # def test_detect_edge_case_empty_text(self):
#     #     """Test the detect method with an empty string."""
#     #     text = ""
#     #     result = self.detector.detect(text=text, return_prediction=True, return_scores=True)
#     #     self.assertIn("prediction", result)
#     #     self.assertFalse(result["prediction"])  # Prediction should be False for empty text

#     def test_detect_with_p_value(self):
#         """Test the detect method with p_value calculation."""
#         text = "A simple test sentence."
#         result = self.detector.detect(text=text, return_p_value=True, return_scores=True)
#         self.assertIn("p_value", result)
#         self.assertIsInstance(result["p_value"], float)
#         self.assertGreaterEqual(result["p_value"], 0)
#         self.assertLessEqual(result["p_value"], 1)

# if __name__ == "__main__":
#     unittest.main()
