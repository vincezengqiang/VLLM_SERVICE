# models/model3_hf.py
from utils.hf_engine import HFEngineWrapper
from config import MODEL_CONFIGS
from utils.logger_local import logger
from typing import Dict, Any, Optional, List

# # 初始化HF引擎
# engine = HFEngineWrapper(MODEL_CONFIGS["model3"])

def generate_model3(prompt: str, generation_params: Optional[Dict[str, Any]] = None) -> str:
    """
    生成文本（同步）
    :param prompt: 输入提示词
    :param generation_params: 生成参数
    :return: 生成的文本
    """
    return engine.generate(prompt, generation_params)

async def generate_model3_async(prompt: str, generation_params: Optional[Dict[str, Any]] = None) -> str:
    """
    异步生成文本
    :param prompt: 输入提示词
    :param generation_params: 生成参数
    :return: 生成的文本
    """
    # 在线程池中运行CPU密集型任务
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_model3, prompt, generation_params)

def batch_generate_model3(prompts: List[str], generation_params: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    批量生成文本（同步）
    :param prompts: 输入提示词列表
    :param generation_params: 生成参数
    :return: 生成的文本列表
    """
    return engine.batch_generate(prompts, generation_params)

async def batch_generate_model3_async(prompts: List[str], generation_params: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    异步批量生成文本
    :param prompts: 输入提示词列表
    :param generation_params: 生成参数
    :return: 生成的文本列表
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, batch_generate_model3, prompts, generation_params)