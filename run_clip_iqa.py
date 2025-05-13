import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment

def run_clip_iqa(image_path):
    # 设置设备
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    metric = CLIPImageQualityAssessment(prompts=(
        'quality', 'brightness', 'noisiness', 'colorfullness', 'sharpness'
    )).to(device)

    # 加载图像并预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        score = metric(img_tensor)

    return score  # 返回 dict， {'quality': 0.79, ...}

# 测试用例
if __name__ == "__main__":
    image_path = "imgs/testdir1/1-1.jpg"
    result = run_clip_iqa(image_path)
    print("CLIP-IQA 分数:")
    for k, v in result.items():
        print(f"{k:15s}: {v:.4f}")
