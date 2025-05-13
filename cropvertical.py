from PIL import Image
import os

def split_image_vertically(image_path):
    # 读取图片
    img = Image.open(image_path)
    width, height = img.size

    # 计算中心
    center_x = width // 2

    # 裁剪左边
    left_img = img.crop((0, 0, center_x, height))
    # 裁剪右边
    right_img = img.crop((center_x, 0, width, height))

    # 构造保存路径
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    left_path = f"{base_name}_left.jpg"
    right_path = f"{base_name}_right.jpg"

    # 保存图片
    left_img.save(left_path)
    right_img.save(right_path)

    print(f"图片已保存为：{left_path}, {right_path}")

# 示例用法：你上传图片后，将路径填入此处
if __name__ == "__main__":
    image_path = "example.png"  # <-- 改成你上传图片的实际文件名
    split_image_vertically(image_path)
