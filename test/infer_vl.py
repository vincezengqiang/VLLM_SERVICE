from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from qwen_vl_utils import process_vision_info

import base64

# default: Load the model on the available device(s)

model_path="/home/work/a03_VLLM_SERVICE/llm_models/Qwen2.5-VL-7B-Instruct"
model_path="/home/work/a03_VLLM_SERVICE/llm_models/Qwen3-VL-8B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
   model_path, torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained(model_path)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


image_path='/home/work/a03_VLLM_SERVICE/阀门.jpg'

"""将图片转换为 base64 编码"""

with open(image_path, "rb") as image_file:
    # 读取图像内容并编码为 Base64
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
data_url = f"data:image/jpeg;base64,{encoded_string}"

prompt="""# 人设
你是一位专业的设备运维与视觉检测专家。请仔细分析用户提供的图像，完成以下两个任务：

# 任务
请仔细观察以下图像，并按以下要求进行分析：
用简洁清晰的语言描述图像中包含的主要内容（例如设备类型、环境、可见组件、屏幕状态、指示灯、接线情况等）。
重点描述图像中的相关数据、指标内容，如温度、压力、流量、电压、电流等。

# 注意：
- 仅当图像中存在清晰、可辨识的视觉证据支持某类异常时，才列出该异常；严禁猜测或过度推断。
- 若未发现异常，请仅描述图像内容，不需要提及没有明显异常。
- 使用自然语言描述，避免使用“1、2”等编号格式。
- 聚焦于可明确识别的内容，忽略模糊、遮挡或无法辨认的细节。
- 不要依次排查可能的异常类型，而是整体观察图像后总结发现的异常。
"""
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": data_url,
            },
            # {
            #     "type": "image",
            #     "image": "/home/work/a03_VLLM_SERVICE/阀门.jpg",
            # },
            {"type": "text", "text": prompt},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)