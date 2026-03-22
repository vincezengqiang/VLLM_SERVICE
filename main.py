# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import time
from config import DEFAULT_GENERATION_CONFIG, SERVER_CONFIG
from utils.logger_local import *
import uvicorn

# 延迟导入 model_manager，避免在模块导入时初始化
model_manager = None

# logger = setup_logger()


# 请求模型定义
class GenerationRequest(BaseModel):
    """单个生成请求"""
    prompt: str
    image: Optional[str] = None
    system: Optional[str] = None
    sampling_params: Optional[Dict[str, Any]] = None


class BatchGenerationRequest(BaseModel):
    """批量生成请求"""
    prompts: List[str]
    sampling_params: Optional[Dict[str, Any]] = None


class MultiModelRequest(BaseModel):
    """多模型生成请求"""
    prompt: str
    sampling_params: Optional[Dict[str, Any]] = None
    models: Optional[List[str]] = None  # 指定要使用的模型，None表示所有模型


# 响应模型定义
class GenerationResponse(BaseModel):
    """生成响应"""
    result: str
    model_used: str
    generation_time: float


class BatchGenerationResponse(BaseModel):
    """批量生成响应"""
    results: List[str]
    model_used: str
    generation_time: float


class MultiModelResponse(BaseModel):
    """多模型生成响应"""
    results: Dict[str, str]
    generation_time: float


class ModelInfoResponse(BaseModel):
    """模型信息响应"""
    models: Dict[str, Dict[str, Any]]


# 创建FastAPI应用
app = FastAPI(title="vLLM Multi-Model API",
              description="支持多个vLLM模型的API服务，支持并发和批量处理",
              version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    global model_manager
    # 延迟初始化模型管理器，避免 multiprocessing 问题
    from models.model_manager import get_model_manager
    model_manager = get_model_manager()
    logger.info("Starting vLLM Multi-Model API server...")
    logger.info(
        f"Server will run on {SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")


@app.get("/")
async def root():
    """根路径"""
    return {"message": "vLLM Multi-Model API Server is running!"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/models/info", response_model=ModelInfoResponse)
async def get_models_info():
    """获取所有模型信息"""
    try:
        info = model_manager.get_model_info()
        return ModelInfoResponse(models=info)
    except Exception as e:
        logger.error(f"Get models info failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/{model_name}", response_model=GenerationResponse)
async def generate_single(model_name: str, request: GenerationRequest):
    """
    单个模型生成
    :param model_name: 模型名称
    :param request: 生成请求
    """
    start_time = time.time()

    # try:
    logger.info(f"model_name: {model_name}")
    # logger.info(f"request: {request}")

    # 合并默认参数和用户参数
    sampling_params = DEFAULT_GENERATION_CONFIG.copy()
    if request.sampling_params:
        sampling_params.update(request.sampling_params)

    if request.image:
        result = await model_manager.generate_single(
            model_name,
            request.prompt,
            system=request.system,
            image=request.image,
            generation_params=sampling_params)
    else:
        result = await model_manager.generate_single(
            model_name,
            request.prompt,
            system=request.system,
            generation_params=sampling_params)
    generation_time = time.time() - start_time

    return GenerationResponse(result=result,
                              model_used=model_name,
                              generation_time=generation_time)
    # except ValueError as e:
    #     raise HTTPException(status_code=404, detail=str(e))
    # except Exception as e:
    #     logger.error(f"Single generation failed for model {model_name}: {str(e)}")
    #     raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch/{model_name}", response_model=BatchGenerationResponse)
async def generate_batch(model_name: str, request: BatchGenerationRequest):
    """
    批量生成
    :param model_name: 模型名称
    :param request: 批量生成请求
    """
    start_time = time.time()

    try:
        # 合并默认参数和用户参数
        sampling_params = DEFAULT_GENERATION_CONFIG.copy()
        if request.sampling_params:
            sampling_params.update(request.sampling_params)

        results = await model_manager.generate_batch(model_name,
                                                     request.prompts,
                                                     sampling_params)
        generation_time = time.time() - start_time

        return BatchGenerationResponse(results=results,
                                       model_used=model_name,
                                       generation_time=generation_time)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            f"Batch generation failed for model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multi-generate", response_model=MultiModelResponse)
async def generate_multi_model(request: MultiModelRequest):
    """
    多模型同时生成
    :param request: 多模型生成请求
    """
    start_time = time.time()

    try:
        # # 合并默认参数和用户参数
        # sampling_params = DEFAULT_GENERATION_CONFIG.copy()
        # if request.sampling_params:
        #     sampling_params.update(request.sampling_params)

        # # # 确定要使用的模型
        # # if request.models:
        # #     # 验证指定的模型是否存在
        # #     available_models = set(model_manager.engines.keys())
        # #     invalid_models = set(request.models) - available_models
        # #     if invalid_models:
        # #         raise ValueError(f"Invalid models: {invalid_models}")
        # #     models_to_use = request.models
        # # else:
        # #     models_to_use = list(model_manager.engines.keys())

        # # 确定要使用的模型
        # if request.models:
        #     # 验证指定的模型是否存在
        #     available_models = set(model_manager.keys())  # 修正：使用 keys() 方法
        #     invalid_models = set(request.models) - available_models
        #     if invalid_models:
        #         raise ValueError(f"Invalid models: {invalid_models}")
        #     models_to_use = request.models
        # else:
        #     models_to_use = list(model_manager.keys())  # 修正：使用 keys() 方法

        # logger.info(f"models_to_use: {models_to_use}")
        # # 并发执行所有模型
        # tasks = {
        #     model_name: model_manager.generate_single(model_name, request.prompt, sampling_params)
        #     for model_name in models_to_use
        # }
        # logger.info(f"tasks: {tasks}")

        # results = {}
        # for model_name, task in tasks.items():
        #     logger.info(f"model_name: {model_name}")
        #     logger.info(f"task: {task}")

        #     try:
        #         results[model_name] = await task
        #     except Exception as e:
        #         logger.error(f"Model {model_name} generation failed: {str(e)}")
        #         results[model_name] = f"Error: {str(e)}"

        # generation_time = time.time() - start_time

        # return MultiModelResponse(
        #     results=results,
        #     generation_time=generation_time
        # )

        # 合并默认参数和用户参数
        sampling_params = DEFAULT_GENERATION_CONFIG.copy()
        if request.sampling_params:
            sampling_params.update(request.sampling_params)

        # 确定要使用的模型（支持重复）
        if request.models:
            # 验证指定的模型是否存在（不去重，保持原始顺序）
            available_models = set(model_manager.keys())
            invalid_models = set(request.models) - available_models
            if invalid_models:
                raise ValueError(f"Invalid models: {invalid_models}")
            models_to_use = request.models  # 保持原始列表，包括重复项
        else:
            models_to_use = list(model_manager.keys())

        # 为重复模型创建唯一标识
        model_tasks = []
        model_count = {}  # 记录每个模型的调用次数

        for model_name in models_to_use:
            # 计算这是第几次调用该模型
            count = model_count.get(model_name, 0) + 1
            model_count[model_name] = count

            # 创建唯一的任务标识
            task_id = f"{model_name}_{count}" if count >= 1 else model_name

            # 创建任务
            task = model_manager.generate_single(model_name, request.prompt,
                                                 sampling_params)
            model_tasks.append((task_id, task, model_name))

        # 并发执行所有任务
        results = {}
        for task_id, task, original_model_name in model_tasks:
            logger.info(f"task_id: {task_id}")
            logger.info(f"task: {task}")
            logger.info(f"original_model_name: {original_model_name}")

            try:
                result = await task
                results[task_id] = result
            except Exception as e:
                logger.error(
                    f"Model {original_model_name} generation failed: {str(e)}")
                results[task_id] = f"Error: {str(e)}"

        generation_time = time.time() - start_time

        return MultiModelResponse(results=results,
                                  generation_time=generation_time)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Multi-model generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 启动服务器
if __name__ == "__main__":
    # uvicorn.run(
    #     "main:app",
    #     host=SERVER_CONFIG["host"],
    #     port=SERVER_CONFIG["port"],
    #     workers=1,  # 单进程
    #     # 并发控制参数
    #     loop="asyncio",  # 使用asyncio事件循环
    #     http="httptools",  # 更快的HTTP解析
    #     limit_concurrency=1000,  # 限制并发连接数
    #     limit_max_requests=10000,  # 限制最大请求数（防内存泄漏）
    #     timeout_keep_alive=5,  # 保持连接超时
    #     backlog=2048,  # socket backlog
    # )
    uvicorn.run(
        "main:app",
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        workers=SERVER_CONFIG["workers"],
        limit_concurrency=SERVER_CONFIG["max_concurrent_requests"],  # 限制并发连接数
        timeout_keep_alive=SERVER_CONFIG["timeout"],
        reload=False  # 生产环境建议设置为False
    )
