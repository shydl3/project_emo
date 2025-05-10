import glob

from PIL import Image
import os
import glob

start_idx = 3
img_dir = os.path.join(os.getcwd(), "raw_imgs")
jpg_files = glob.glob(os.path.join(img_dir, "*.png"))
jpg_files.sort(key=lambda x: os.path.basename(x))
# print(jpg_files)

crop_box = (0, 350, 1180, 1920)


for i in range(0, len(jpg_files) // 2):
    pair = jpg_files[2*i:2*i+2]
    # print(pair)
    for j, img_path in enumerate(pair, 1):
        # print(j)
        img = Image.open(img_path)
        cropped_img = img.crop(crop_box)
        cropped_img.save(f"./cropped_imgs/{start_idx}-{j}.jpg")

    start_idx += 1
print("done")
