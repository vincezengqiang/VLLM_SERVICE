# models/model_manager.py
from typing import Dict, Any, Optional, List
from utils.vllm_engine import VLLMEngineWrapper
from utils.hf_engine import HFEngineWrapper
# from models.model3_hf import generate_model3_async, batch_generate_model3_async
from config import MODEL_CONFIGS
from utils.logger_local import logger
import asyncio


class ModelManager:
    """模型管理器"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.vllm_engines: Dict[str, VLLMEngineWrapper] = {}
            self.hf_engines: Dict[str, HFEngineWrapper] = {}
            self.model_configs = MODEL_CONFIGS  # 保存模型配置
            self._load_models()
            ModelManager._initialized = True

    def _load_models(self):
        """加载所有模型"""
        logger.info("Loading models...")
        for model_name, config in MODEL_CONFIGS.items():
            logger.info(f"Loading model {model_name} with config: {config}")
            try:
                if config["type"] == "vllm":
                    engine = VLLMEngineWrapper(config)
                    self.vllm_engines[model_name] = engine
                    logger.info(f"vLLM Model {model_name} loaded successfully")
                elif config["type"] == "huggingface":
                    # 初始化HF引擎
                    engine = HFEngineWrapper(config)
                    self.hf_engines[model_name] = engine
                    # HF模型在导入时已经初始化
                    logger.info(
                        f"HF Model {model_name} will be loaded on demand")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                raise

    async def generate_single(self,
                              model_name: str,
                              prompt: str,
                              system: Optional[str] = None,
                              image: Optional[str] = None,
                              generation_params: Optional[Dict[str, Any]] = None) -> str:
        """
        单个模型生成
        :param model_name: 模型名称
        :param prompt: 提示词
        :param generation_params: 生成参数
        :return: 生成结果
        """
        logger.info(f"model_name :{model_name}")
        logger.info(f"prompt :{prompt}")
        # logger.info(f"image :{image}")
        logger.info(f"generation_params :{generation_params}")

        if model_name not in MODEL_CONFIGS:
            logger.info(f"Model {model_name} not found")
            raise ValueError(f"Model {model_name} not found")

        model_config = MODEL_CONFIGS[model_name]

        if model_config["type"] == "vllm":
            if model_name not in self.vllm_engines:
                raise ValueError(
                    f"vLLM Model {model_name} not properly loaded")
            return await self.vllm_engines[model_name].generate(prompt, system, image, generation_params)
        elif model_config["type"] == "huggingface":
            if model_name == "model3":
                return await self.hf_engines[model_name].generate(prompt, generation_params)
            else:
                raise ValueError(f"Unsupported HF model: {model_name}")
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")

    async def generate_batch(self,
                             model_name: str,
                             prompts: List[str],
                             generation_params: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        批量生成
        :param model_name: 模型名称
        :param prompts: 提示词列表
        :param generation_params: 生成参数
        :return: 生成结果列表
        """
        logger.info(f"model_name :{model_name}")
        logger.info(f"prompt :{prompts}")
        logger.info(f"generation_params :{generation_params}")

        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} not found")

        model_config = MODEL_CONFIGS[model_name]

        if model_config["type"] == "vllm":
            if model_name not in self.vllm_engines:
                raise ValueError(
                    f"vLLM Model {model_name} not properly loaded")
            return await self.vllm_engines[model_name].batch_generate(prompts, generation_params)
        elif model_config["type"] == "huggingface":
            if model_name == "model3":
                return await self.hf_engines[model_name].batch_generate(prompts, generation_params)
                raise ValueError(f"Unsupported HF model: {model_name}")
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")

    async def generate_all(self,
                           prompt: str,
                           generation_params: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        所有模型同时生成
        :param prompt: 提示词
        :param generation_params: 生成参数
        :return: 各模型生成结果字典
        """
        tasks = {}
        for model_name in MODEL_CONFIGS.keys():
            tasks[model_name] = self.generate_single(
                model_name, prompt, generation_params)

        results = {}
        for model_name, task in tasks.items():
            try:
                results[model_name] = await task
            except Exception as e:
                logger.error(f"Model {model_name} generation failed: {str(e)}")
                results[model_name] = f"Error: {str(e)}"

        return results

    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """获取模型信息"""
        info = {}
        for name, config in MODEL_CONFIGS.items():
            model_info = {
                "type": config["type"],
                "path": config["path"],
            }
            if config["type"] == "vllm":
                model_info.update({
                    "device": config["device"],
                    "gpu_memory_utilization": config["gpu_memory_utilization"],
                })
            elif config["type"] == "huggingface":
                model_info.update({
                    "devices": config.get("devices", []),
                    "max_memory": config.get("max_memory", {}),
                })
            info[name] = model_info
        return info

    def keys(self):
        """返回所有模型名称"""
        return self.model_configs.keys()

    def __contains__(self, model_name):
        """检查模型是否存在"""
        return model_name in self.model_configs


# 全局模型管理器实例（延迟初始化）
model_manager = None

def get_model_manager():
    """获取模型管理器实例（延迟初始化）"""
    global model_manager
    if model_manager is None:
        model_manager = ModelManager()
    return model_manager
