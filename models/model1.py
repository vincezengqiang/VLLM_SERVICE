# models/model1.py
from utils.vllm_engine import create_vllm_engine

# 假设模型1使用 GPU 0
engine = create_vllm_engine("/path/to/model1", gpu_memory_utilization=0.8, device="cuda:0")

async def generate_model1(prompt: str, max_tokens: int = 50):
    results_generator = engine.generate(prompt, sampling_params=None, request_id="model1")
    async for request_output in results_generator:
        pass
    return request_output.outputs[0].text