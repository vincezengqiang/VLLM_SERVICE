# utils/vllm_engine.py
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncEngineArgs, AsyncLLMEngine
from typing import Dict, Any, Optional, List
import asyncio
from utils.logger_local import logger
import os
import base64
import os
import base64
from PIL import Image
from io import BytesIO
from transformers import AutoTokenizer


class VLLMEngineWrapper:
    """vLLM引擎包装器"""

    def __init__(self, model_config: Dict[str, Any]):
        """
        初始化vLLM引擎
        :param model_config: 模型配置字典
        """
        self.model_config = model_config
        self.engine = self._create_engine()
        self.tokenizer = AutoTokenizer.from_pretrained(model_config["path"])

        logger.info(f"Engine created for model: {model_config['path']}")

    def _create_engine(self) -> AsyncLLMEngine:
        logger.info(f"model_config: {self.model_config}")

        if 'gpu_id' in self.model_config:
            gpu_id = self.model_config['gpu_id']
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            logger.info(f"Setting CUDA_VISIBLE_DEVICES={gpu_id} for model")
        """创建vLLM引擎实例"""
        engine_args = AsyncEngineArgs(
            model=self.model_config["path"],
            gpu_memory_utilization=self.model_config["gpu_memory_utilization"],
            max_model_len=self.model_config.get("max_model_len", 4096),
            trust_remote_code=True,  # 如果需要加载自定义模型
            tensor_parallel_size=self.model_config["tp_size"],  # 如果需要加载自定义模型
            # vLLM并发配置
            max_num_batched_tokens=self.model_config.get(
                "max_num_batched_tokens", 4096),
            max_num_seqs=self.model_config.get("max_num_seqs", 2),
        )

        return AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(self,
                       prompt: str,
                       system: Optional[str] = None,
                       image: Optional[str] = None,
                       sampling_params: Optional[Dict[str, Any]] = None,
                       request_id: Optional[str] = None) -> str:
        """
        生成文本
        :param prompt: 输入提示词
        :param sampling_params: 采样参数
        :param request_id: 请求ID
        :return: 生成的文本
        """
        # try:
        # 创建采样参数
        if sampling_params is None:
            sampling_params = {}

        params = SamplingParams(
            temperature=sampling_params.get("temperature", 0.7),
            top_p=sampling_params.get("top_p", 0.9),
            max_tokens=sampling_params.get("max_tokens", 512),
            stop=sampling_params.get("stop"),
            presence_penalty=sampling_params.get("presence_penalty", 0.0),
            frequency_penalty=sampling_params.get("frequency_penalty", 0.0),
        )
        enable_thinking = sampling_params.get("enable_thinking", False)
        # 生成唯一请求ID
        if request_id is None:
            request_id = f"req_{hash(prompt) % 1000000}"

        if system != None:
            messages = [{
                "role": "system",
                "content": system
            }, {
                "role": "user",
                "content": prompt
            }]
        else:
            messages = [{"role": "user", "content": prompt}]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking)

        if image != None:

            def load_image(encoded_string, max_size=(1024, 1024)):
                if "," in encoded_string:
                    header, encoded = encoded_string.split(",", 1)
                else:
                    encoded = encoded_string  # 如果没有前缀，就直接用

                # 1. 解码 Base64 字符串为字节
                image_data = base64.b64decode(encoded)

                # 2. 将字节数据转为图像对象
                image = Image.open(BytesIO(image_data))

                # 3. 等比例缩放图像
                image.thumbnail(max_size)

                return image

            image = load_image(image)

            # 构造多模态输入
            messages = [{
                "role":
                "user",
                "content": [{
                    "type": "image"
                }, {
                    "type": "text",
                    "text": prompt
                }]
            }]
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking)

            inputs = {
                "prompt": input_text,
                "multi_modal_data": {
                    "image": image  # 或者 image_path，取决于 vLLM 实现
                }
            }
        logger.info(f"inputs: {inputs}")

        # 执行生成
        results_generator = self.engine.generate(inputs,
                                                 params,
                                                 request_id=request_id)

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output and final_output.outputs:
            logger.info(
                f"final_output.outputs[0].text: {final_output.outputs[0].text}"
            )
            return final_output.outputs[0].text
        else:
            logger.info(f"final_output.outputs[0].text: 无结果")
            return ""

        # except Exception as e:
        #     logger.error(f"Generation failed: {str(e)}")
        #     raise

    async def batch_generate(
            self,
            prompts: List[str],
            sampling_params: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        批量生成文本
        :param prompts: 输入提示词列表
        :param sampling_params: 采样参数
        :return: 生成的文本列表
        """
        try:
            tasks = [
                self.generate(prompt, sampling_params, f"batch_req_{i}")
                for i, prompt in enumerate(prompts)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理异常结果
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch generation error: {str(result)}")
                    processed_results.append("")
                else:
                    processed_results.append(result)

            return processed_results

        except Exception as e:
            logger.error(f"Batch generation failed: {str(e)}")
            raise
