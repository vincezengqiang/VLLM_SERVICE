# config.py
from typing import Dict, Any

# 模型配置
MODEL_CONFIGS = {
    "model1": {
        "type": "vllm",
        # "path": "./llm_models/Qwen3-8B",  # 替换为实际模型路径
        # "path": "./llm_models/Qwen2.5-VL-7B-Instruct",  # 替换为实际模型路径
        # "path": "./llm_models/Qwen3-VL-8B-Instruct",  # 替换为实际模型路径
        # "path": "./llm_models/Qwen3.5-4B",  # 替换为实际模型路径
        "path": "./llm_models/Qwen3.5-9B",  # 替换为实际模型路径
        "gpu_id": 0,  # 指定使用GPU 0
        "gpu_memory_utilization": 0.95,
        "max_model_len": 12288,
        "tp_size": 1,  # 替换为你的模型的 tensor parallel size
        "max_num_seqs": 1,  # 最大并发序列数
        "max_num_batched_tokens": 1024,  # 最大批处理令牌数
        "swap_space": 4, 
        "enforce_eager": True, 
        "disable_log_stats": False, 
    },
    # "model2": {
    #     "type": "vllm",
    #     "path": "/path/to/your/model2",  # 替换为实际模型路径
    #     "gpu_id": 1,  # 指定使用GPU 1
    #     "gpu_memory_utilization": 0.8,
    #     "max_model_len": 4096,
    # },
    # "model3": {
    #     "type": "huggingface",
    #     "path": "/path/to/your/model3",  # 替换为实际模型路径
    #     "devices": ["cuda:0", "cuda:1"],  # 使用两张GPU
    #     "max_memory": {0: "4GB", 1: "4GB"},  # 每张卡分配4GB显存（约20%）
    #     "load_in_8bit": False,  # 是否使用8bit量化节省显存
    #     "torch_dtype": "float16",  # 数据类型
    # }
}

# 默认生成参数
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0,
    "top_p": 0.9,
    "max_tokens": 1024,
    "stop": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "enable_thinking": False,
}

# 服务器配置
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 1,
    "max_concurrent_requests": 100,  # 最大并发请求数
    "timeout": 30,  # 请求超时时间
}
