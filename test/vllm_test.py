from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.engine.async_llm_engine import AsyncEngineArgs, AsyncLLMEngine

import os

os.environ["cuda_visible_devices"] = "0"  # 设置可见的 GPU 设备

# 模型名称
model_name = "./llm_models/Qwen3-8B"  # 替换为你的模型名称

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_model_len = 8192  # 替换为你的模型最大长度
tp_size = 1  # 替换为你的模型的 tensor parallel size


# engine_args = AsyncEngineArgs(
#             model=model_name,
#             gpu_memory_utilization=0.8,        
#             max_model_len=4096,
#             trust_remote_code=True,  # 如果需要加载自定义模型
#             tensor_parallel_size=tp_size,  # 如果需要加载自定义模型
#         )
# AsyncLLMEngine.from_engine_args(engine_args)
# print('done')

# 加载模型
model = LLM(model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=tp_size,  # 替换为你的模型的 tensor parallel size
            trust_remote_code=True,  # 如果需要信任远程代码
            enforce_eager=True,  # 强制使用 eager 模式
            enable_chunked_prefill=True,  # 启用分块处理
            max_num_batched_tokens=1024,  # 设置批处理的最大 token 数
            gpu_memory_utilization=0.8)  # 替换为你的显存利用率

# 输入文本
messages=[{"role": "user", "content": "你好，世界！"}]
# 推理参数
stop_token_id=[151645,151643]
sampling_params = SamplingParams(temperature=0, max_tokens=512,stop_token_ids=stop_token_id)

# 使用 tokenizer 对输入文本进行编码
# inputs = tokenizer(input_text, return_tensors="pt")
inputs=tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True,enable_thinking=False)

# 执行前向传播
outputs = model.generate(inputs,sampling_params)

# 使用 tokenizer 解码输出
generated_text = outputs[0].outputs[0].text

# 打印结果
print("Input Text:", messages)
print("Generated Output:", generated_text)