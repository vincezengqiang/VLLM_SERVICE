import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"  # 根据实际类型修改 image/jpeg

# 示例
base64_str = image_to_base64("/home/work/a03_VLLM_SERVICE/阀门.jpg")
print(base64_str[:100] + "...")  # 打印前100字符