import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment

# python clip_iqa_demo.py
# 设置设备（macOS 下为 CPU 或 MPS）
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 加载模型
metric = CLIPImageQualityAssessment(prompts=('quality', 'brightness', 'noisiness', 'colorfullness', 'sharpness', )).to(device)

# 加载图像并预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img = Image.open("imgs/1-1.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# 推理
with torch.no_grad():
    score = metric(img_tensor)
    print("CLIP-IQA 分数:")
    for item in score.items():
        print(item)

