from PIL import Image
import os

def split_image_vertically(image_path):
    # 读取图片
    img = Image.open(image_path)
    width, height = img.size

    # 计算中心
    center_y = height // 2

    top_img = img.crop((0,0, width, center_y))

    bottom_img = img.crop((0, center_y, width, height))

    # 构造保存路径
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    top_path = f"{base_name}_top.jpg"
    bottom_path = f"{base_name}_bottom.jpg"

    # 保存图片
    top_img.save(top_path)
    bottom_img.save(bottom_path)

    print(f"图片已保存为：{top_path}, {bottom_path}")

# 示例用法：你上传图片后，将路径填入此处
if __name__ == "__main__":
    image_path = "example.png"  # <-- 改成你上传图片的实际文件名
    split_image_vertically(image_path)
