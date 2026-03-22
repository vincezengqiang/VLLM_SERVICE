# utils/hf_engine.py
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from typing import Dict, Any, Optional, List
import asyncio
from utils.logger_local import logger

class HFEngineWrapper:
    """HuggingFace引擎包装器"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        初始化HuggingFace引擎
        :param model_config: 模型配置字典
        """
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self._load_model()
        logger.info(f"HF Engine created for model: {model_config['path']}")
    
    def _load_model(self):
        """加载HuggingFace模型"""
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["path"],
                trust_remote_code=True
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型配置
            torch_dtype = getattr(torch, self.model_config.get("torch_dtype", "float16"))
            
            # 加载模型
            load_kwargs = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": True,
            }
            
            # 如果启用8bit量化
            if self.model_config.get("load_in_8bit", False):
                load_kwargs["load_in_8bit"] = True
            else:
                # 显存分配配置
                if "max_memory" in self.model_config:
                    load_kwargs["max_memory"] = self.model_config["max_memory"]
                if "devices" in self.model_config:
                    load_kwargs["device_map"] = "auto"  # 自动分配到多GPU
            
            logger.info(f"Loading HF model with config: {load_kwargs}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config["path"],
                **load_kwargs
            )
            
            # 如果没有使用device_map，手动移动到指定设备
            if "device_map" not in load_kwargs:
                devices = self.model_config.get("devices", ["cuda:0"])
                self.model = self.model.to(devices[0])
            
            self.model.eval()
            logger.info("HF model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load HF model: {str(e)}")
            raise
    
    def generate(self, 
                prompt: str, 
                generation_params: Optional[Dict[str, Any]] = None) -> str:
        """
        生成文本
        :param prompt: 输入提示词
        :param generation_params: 生成参数
        :return: 生成的文本
        """
        try:
            if generation_params is None:
                generation_params = {}
            
            # 编码输入
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                # max_length=2048
            )
            
            # 移动到模型设备
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 创建生成配置
            gen_config = GenerationConfig(
                temperature=generation_params.get("temperature", 0.7),
                top_p=generation_params.get("top_p", 0.9),
                max_new_tokens=generation_params.get("max_tokens", 512),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text
            
        except Exception as e:
            logger.error(f"HF generation failed: {str(e)}")
            raise

    def batch_generate(self, 
                      prompts: List[str], 
                      generation_params: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        批量生成文本
        :param prompts: 输入提示词列表
        :param generation_params: 生成参数
        :return: 生成的文本列表
        """
        try:
            if generation_params is None:
                generation_params = {}
            
            # 批量编码输入
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # 移动到模型设备
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 创建生成配置
            gen_config = GenerationConfig(
                temperature=generation_params.get("temperature", 0.7),
                top_p=generation_params.get("top_p", 0.9),
                max_new_tokens=generation_params.get("max_tokens", 512),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                )
            
            # 解码输出
            generated_texts = []
            for i, output in enumerate(outputs):
                input_length = inputs["input_ids"][i].shape[0]
                generated_text = self.tokenizer.decode(
                    output[input_length:], 
                    skip_special_tokens=True
                )
                generated_texts.append(generated_text)
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"HF batch generation failed: {str(e)}")
            raise