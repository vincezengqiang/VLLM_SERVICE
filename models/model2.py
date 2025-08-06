# models/model2.py
from utils.vllm_engine import create_vllm_engine

# 假设模型2使用 GPU 1
engine = create_vllm_engine("/path/to/model2", gpu_memory_utilization=0.8, device="cuda:1")

async def generate_model2(prompt: str, max_tokens: int = 50):
    results_generator = engine.generate(prompt, sampling_params=None, request_id="model2")
    async for request_output in results_generator:
        pass
    return request_output.outputs[0].text