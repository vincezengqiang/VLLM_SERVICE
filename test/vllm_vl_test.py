from vllm import LLM, SamplingParams
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoTokenizer
import os,base64

def load_image(image_path_or_url):
    """加载图像"""
    if image_path_or_url.startswith('http'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_url)
    image.thumbnail((1024, 1024))
    return image

# 初始化模型
# 注意：需要根据实际模型路径调整
model_path = "/home/work/a03_VLLM_SERVICE/llm_models/Qwen2.5-VL-7B-Instruct"  # 或你的本地模型路径

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    # dtype="half",  # 使用半精度以节省内存
    max_model_len=4096,  # 根据需要调整
    # 如果有多个GPU，可以指定tensor_parallel_size
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,        
    # limit_mm_per_prompt={"image": 1500*28*28}
    limit_mm_per_prompt={"image": 1024*1024}
)

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    repetition_penalty=1.05,
    max_tokens=1024
)

# # 示例1：纯文本推理
# text_only_prompt = "你好，请介绍一下你自己。"
# messages = [
#     {"role": "user", "content": text_only_prompt}
# ]

# # 转换为模型所需的格式
# prompt = llm.get_tokenizer().apply_chat_template(
#     messages, 
#     tokenize=False, 
#     add_generation_prompt=True
# )

# outputs = llm.generate([prompt], sampling_params)
# print("纯文本推理结果:")
# print(outputs[0].outputs[0].text)
# print("-" * 50)

# 示例2：图文推理
# 加载图像
image_path = "/home/work/a03_VLLM_SERVICE/阀门.jpg"  # 替换为实际图像路径
# 或者使用URL: image_url = "https://example.com/image.jpg"


image = load_image(image_path)

# 构造多模态输入
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "请描述这张图片的内容。"}
        ]
    }
]

prompt = llm.get_tokenizer().apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

print(prompt)
inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "image": image  # 或者 image_path，取决于 vLLM 实现
    }
}


# 生成时传入图像
outputs = llm.generate(
    inputs, 
    sampling_params,
)
# outputs = llm.generate(
#     [prompt], 
#     sampling_params,
#     multi_modal_data=[{"image": image}]
# )

print("图文推理结果:")
print(outputs[0].outputs[0].text)




# def resize_base64_image(encoded_string, max_size=(1024, 1024)):
#     if "," in encoded_string:
#         header, encoded = data_url.split(",", 1)
#     else:
#         encoded = data_url  # 如果没有前缀，就直接用

#     # 1. 解码 Base64 字符串为字节
#     image_data = base64.b64decode(encoded)

#     # 2. 将字节数据转为图像对象
#     image = Image.open(BytesIO(image_data))

#     # 3. 等比例缩放图像
#     image.thumbnail(max_size)

#     # 4. 将缩放后的图像保存到内存
#     buffered = BytesIO()
#     image.save(buffered, format=image.format or "JPEG")

#     # 5. 重新编码为 Base64
#     resized_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

#     # 6. 返回 Base64 数据 URL 格式
#     mime_type = "image/png" if image.format == "PNG" else "image/jpeg"
#     # return f"{mime_type};base64,{resized_base64}"
#     return resized_base64

# data_url1=resize_base64_image(data_url)