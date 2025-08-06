from vllm import LLM, SamplingParams
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoTokenizer

model_path = "/home/work/a03_VLLM_SERVICE/llm_models/Qwen2.5-VL-7B-Instruct"  # 或你的本地模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "请描述这张图片的内容。"}
        ]
    }
]
messages=[{"role": "user", "content": "描述一下图像"}]

prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
print(prompt)
