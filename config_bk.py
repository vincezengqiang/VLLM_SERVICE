# config.py
from typing import Dict, Any

# 模型配置
MODEL_CONFIGS = {
    "model1": {
        "type": "vllm",
        "path": "/path/to/your/model1",  # 替换为实际模型路径
        "device": "cuda:0",
        "gpu_memory_utilization": 0.8,
        "max_model_len": 4096,
    },
    "model2": {
        "type": "vllm",
        "path": "/path/to/your/model2",  # 替换为实际模型路径
        "device": "cuda:1",
        "gpu_memory_utilization": 0.8,
        "max_model_len": 4096,
    },
    "model3": {
        "type": "huggingface",
        "path": "/path/to/your/model3",  # 替换为实际模型路径
        "devices": ["cuda:0", "cuda:1"],  # 使用两张GPU
        "max_memory": {0: "4GB", 1: "4GB"},  # 每张卡分配4GB显存（约20%）
        "load_in_8bit": False,  # 是否使用8bit量化节省显存
        "torch_dtype": "float16",  # 数据类型
    }
}

# 默认生成参数
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "stop": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}

# 服务器配置
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 2,
}