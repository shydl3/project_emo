import yaml
from collections import defaultdict
from pathlib import Path

# 定义三个路径
files = [
    "/mnt/data/clipiqa.yml",
    "/mnt/data/nima.yml",
    "/mnt/data/prompt_match.yml"
]

# 初始化合并结构
merged_env = {
    "name": "project_emo",
    "channels": [],
    "dependencies": []
}
pip_packages = set()
conda_packages = set()
channels_set = set()

for file in files:
    with open(file, 'r') as f:
        data = yaml.safe_load(f)

        # 合并 channels
        for ch in data.get("channels", []):
            channels_set.add(ch)

        # 合并 dependencies
        for dep in data.get("dependencies", []):
            if isinstance(dep, str):
                conda_packages.add(dep)
            elif isinstance(dep, dict) and "pip" in dep:
                pip_packages.update(dep["pip"])

# 设置最终内容
merged_env["channels"] = sorted(channels_set)
merged_env["dependencies"] = sorted(conda_packages)
if pip_packages:
    merged_env["dependencies"].append({"pip": sorted(pip_packages)})

# 写入输出文件
output_path = "/mnt/data/project_emo.yml"
with open(output_path, "w") as f:
    yaml.dump(merged_env, f, sort_keys=False)

output_path
