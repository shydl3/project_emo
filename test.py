import os
import glob

img_dir = os.path.join(os.getcwd(), "imgs")

dir_list = [f for f in os.listdir(img_dir)
        if os.path.isdir(os.path.join(img_dir, f))]

for sub_dir in dir_list:
        full_dir = os.path.join(img_dir, sub_dir)
        jpg_files = glob.glob(os.path.join(full_dir, "*.jpg"))
        jpg_files.sort(key=lambda  x: os.path.basename(x))

        # print(jpg_files)


