# test_concurrency_config.py
import asyncio
import aiohttp
import time
# 测试并发
async def find_optimal_concurrency():
    """找到最优并发配置"""
    results = []
    
    # 测试不同并发数
    for concurrency in [50, 100, 200, 300, 500]:
        print(f"\n测试并发数: {concurrency}")
        
        start_time = time.time()
        success_count = 0
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(concurrency):
                task = session.post(
                    "http://localhost:8000/generate/model1",
                    json={"prompt": f"Test {i}", "sampling_params": {"max_tokens": 50}}
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 统计成功数
            for response in responses:
                if not isinstance(response, Exception):
                    if hasattr(response, 'status') and response.status == 200:
                        success_count += 1
        
        total_time = time.time() - start_time
        qps = concurrency / total_time if total_time > 0 else 0
        
        results.append({
            'concurrency': concurrency,
            'success_rate': success_count / concurrency * 100,
            'qps': qps,
            'avg_response_time': total_time / concurrency * 1000
        })
        
        print(f"成功率: {success_count}/{concurrency} ({success_count/concurrency*100:.1f}%)")
        print(f"QPS: {qps:.2f}")
        print(f"平均响应时间: {total_time/concurrency*1000:.1f}ms")
    
    return results

# 运行测试
# results = asyncio.run(find_optimal_concurrency())