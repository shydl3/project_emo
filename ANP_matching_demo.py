import torch
import clip
from PIL import Image
import re

# python ANP_matching_demo.py
# Step 1: 提取 ANP 描述短语（从 txt 文件）
anp_file = "./3244ANPs.txt"  # 替换为你的路径
anp_list = []
with open(anp_file, "r", encoding="utf-8") as f:
    for line in f:
        match = re.match(r"\s+([a-z_]+) \[sentiment", line)
        if match:
            anp_list.append(match.group(1).replace("_", " "))

# Step 2: 加载模型和图像
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("当前使用设备：", device)
image_path = "imgs/testdir1/1-1.jpg"
model, preprocess = clip.load("ViT-B/32", device=device)
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Step 3: 批量处理文本，计算相似度
batch_size = 256
scores = []

with torch.no_grad():
    image_features = model.encode_image(image)
    # image_features /= image_features.norm(dim=-1, keepdim=True) # 取消注释，启用归一化

    for i in range(0, len(anp_list), batch_size):
        batch_prompts = anp_list[i:i+batch_size]
        text_tokens = clip.tokenize(batch_prompts).to(device)
        text_features = model.encode_text(text_tokens)
        # text_features /= text_features.norm(dim=-1, keepdim=True) # 取消注释，启用归一化

        similarity = (image_features @ text_features.T).squeeze(0)
        for j, score in enumerate(similarity):
            scores.append((batch_prompts[j], score.item()))

# Step 4: 输出 top-10 匹配
scores.sort(key=lambda x: x[1], reverse=True)
print(f"\n图片: {image_path}")
print("\nTop 10 最匹配的 ANP：\n")
for desc, score in scores[:10]:
    print(f"{desc:30s} → score: {score:.4f}")
