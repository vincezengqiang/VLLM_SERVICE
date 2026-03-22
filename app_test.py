import requests
import json
import base64


def image_to_base64(image_path):
    """将图像转换为 base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_vllm_api():
    # API 端点
    url = "http://localhost:8000/generate/model1"

    # 请求头
    headers = {
        "Content-Type": "application/json"
    }

    # 请求数据
    data = {
        "prompt": "介绍一下马斯克",
        "sampling_params": {
            "temperature": 0,
            "max_tokens": 128
        }
    }

    try:
        # 发送 POST 请求
        response = requests.post(
            url=url,
            headers=headers,
            json=data,  # 使用 json 参数自动处理序列化
            timeout=30  # 设置超时时间
        )

        # 检查响应状态
        response.raise_for_status()  # 如果状态码不是 2xx，会抛出异常

        # 解析响应
        result = response.json()
        print("响应结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        return result

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
        print(f"响应内容: {response.text}")
        return None


def call_multimodal_vllm_api():
    # API 端点
    url = "http://localhost:8000/generate/model1"

    # 请求头
    headers = {
        "Content-Type": "application/json"
    }

    # 如果需要发送图像，先转换为 base64
    image_base64 = image_to_base64("/home/work/a03_VLLM_SERVICE/阀门.jpg")

    # 多模态请求数据示例
    data = {
        "prompt": "请描述这张图片的内容",  # 或者使用多模态格式
        "sampling_params": {
            "temperature": 0.7,
            "max_tokens": 512
        },
        # 如果 API 支持多模态，可能需要额外参数：
        "image": image_base64  # 具体格式取决于 API 实现
    }

    try:
        # 发送 POST 请求
        response = requests.post(
            url=url,
            headers=headers,
            json=data,
            timeout=60  # 多模态请求可能需要更长时间
        )

        # 检查响应状态
        response.raise_for_status()

        # 解析响应
        result = response.json()
        print("响应结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        return result

    except requests.exceptions.Timeout:
        print("请求超时")
    except requests.exceptions.ConnectionError:
        print("连接错误，请检查服务器是否运行")
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
        print(f"响应内容: {response.text}")

    return None


# 调用函数
if __name__ == "__main__":
    # result = call_vllm_api()  # 普通文本请求
    result = call_multimodal_vllm_api()  # 多模态请求
