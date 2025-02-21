from flask import Flask, render_template, request, jsonify
import os
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList
import sys
sys.path.append("/home/shenhm/documents/lm-watermarking/watermark_reliability_release")
from watermark_processor import WatermarkLogitsProcessor,WatermarkDetector
from watermark_processor_kwg import WatermarkLogitsProcessor as KWG_WatermarkLogitsProcessor
import gc
import torch


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tokenizer = AutoTokenizer.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("/home/shenhm/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",local_files_only=True,device_map = "auto")
device =model.device


app = Flask(__name__, static_folder='assets')
# 设置文件上传路径
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'docx', 'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 判断文件类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')  # 渲染前端HTML页面

# # 为 calendar.html 创建视图函数
# @app.route('/index')
# def index1():
#     return render_template('index.html')

# 为 calendar.html 创建视图函数
@app.route('/watermark_generation.html')
def generation():
    return render_template('watermark_generation.html')

@app.route('/watermark_detection.html')
def detection():
    return render_template('watermark_detection.html')

@app.route('/generate_text', methods=['POST'])
def generate_text():
    # 获取前端传来的JSON数据
    data = request.get_json()
    prompt = data.get('prompt')
    model_name = data.get('model', 'text-davinci-003')  # 默认模型
    max_tokens = data.get('max_tokens', 150)
    watermark_method = data.get('watermark_method', 'simple')
    delta = data.get('delta', 0.5)
    gamma = data.get('gamma', 0.5)


    input_ids = tokenizer.encode(prompt, return_tensors="pt")[:,1:].to(device)
    prefix_len = input_ids.shape[1]

    watermark_processor = WatermarkLogitsProcessor(prefix_len = prefix_len,
                                                    vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=gamma,
                                                    delta=delta,)

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    try:
        # 调用OpenAI API进行文本生成
        # response = openai.Completion.create(
        #     engine="text-davinci-003",  # 你可以选择不同的语言模型
        #     prompt=prompt,
        #     max_tokens=150  # 你可以根据需要调整生成的长度
        # )

        # 获取生成的文本
        output_w = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            num_beams=1,
            do_sample=False,
            repetition_penalty=1.2,
            logits_processor=LogitsProcessorList([watermark_processor]),
            temperature=None,  # ✅ 取消 temperature
            top_p=None  # ✅ 取消 top_p
        )
        generated_text = tokenizer.decode(output_w[0,prefix_len:], skip_special_tokens=True)
        #generated_text = "The dig proved difficult because the area was waterlogged due, in part, to the recent completion of the Grand Union Canal (which was constructed over the original townsite) , although the main problem lay elsewhere; the entire area was being farmed, and therefore had many ploughmarks across the field, making any finds extremely difficult to locate . As a result much time was spent with the helpers removing earth from the fields through the winter period, and then carefully sorting it back into different categories depending whether or no iron tools etc could be seen amongst the mud . In spite all their hard work however , very little was uncover"#response.choices[0].text.strip()
        print(generated_text)
        # 删除模型对象
        # del model

        # # 清理显存
        # torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()

        # 运行 Python 的垃圾回收
        # gc.collect()
        detector = WatermarkDetector(
            threshold_len=0,
            gamma=gamma,
            delta=delta,
            device=device,
            tokenizer=tokenizer,
            vocab=list(tokenizer.get_vocab().values()),
            z_threshold=4.0,
            normalizers=["unicode"],
            ignore_repeated_ngrams=False,
        )
        detector.visual = True

        result = detector.detect(text=generated_text )
        z_score = result['z_score'] #calculate_z_score(generated_text)
        text_length = result['num_tokens_scored']  #len(generated_text.split())
        green_percentage = result['green_fraction']  #(sum(watermark_flags) / text_length) * 100
        p_values = result['p_value']
        watermark_flags = result['green_token_mask']

        # 返回生成的文本和分析信息
        return jsonify({
            'generated_text': generated_text,
            'watermark_flags': watermark_flags,
            'z_score': z_score,
            'text_length': text_length,
            'green_percentage': green_percentage
        })

    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

# 文件上传处理接口
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        # 模拟处理文件
        result = process_text_file(filename)
        return jsonify({"message": "File uploaded and processed", "data": result}), 200
    else:
        return jsonify({"error": "Invalid file type"}), 400

# 模拟文本处理函数
def process_text_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    # 假设我们处理文本并返回字符数作为示例
    return {"word_count": len(text.split()), "char_count": len(text)}

if __name__ == '__main__':
    # app = Flask(__name__, static_folder='assets')
    app.run(debug=True)
