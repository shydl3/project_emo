import torch
import clip
from PIL import Image
import re
import os


def match_anp_with_image(image_path, anp_file="./3244ANPs.txt", model_name="ViT-B/32", batch_size=256, top_k=10):
    # Step 1: 读取 ANP 描述短语
    anp_list = []
    with open(anp_file, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"\s+([a-z_]+) \[sentiment", line)
            if match:
                anp_list.append(match.group(1).replace("_", " "))

    # Step 2: 加载模型和图像
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Step 3: 计算图像与 ANP 的相似度
    scores = []
    with torch.no_grad():
        image_features = model.encode_image(image)
        # image_features /= image_features.norm(dim=-1, keepdim=True)  # 是否归一化，可视情况启用

        for i in range(0, len(anp_list), batch_size):
            batch_prompts = anp_list[i:i+batch_size]
            text_tokens = clip.tokenize(batch_prompts).to(device)
            text_features = model.encode_text(text_tokens)
            # text_features /= text_features.norm(dim=-1, keepdim=True)  # 是否归一化，可视情况启用

            similarity = (image_features @ text_features.T).squeeze(0)
            for j, score in enumerate(similarity):
                scores.append((batch_prompts[j], score.item()))

    # Step 4: 排序并返回 top-k 匹配
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]