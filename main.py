import os
import glob


from run_anp_matching import match_anp_with_image
from run_clip_iqa import run_clip_iqa
from nima.evaluate_inception_resnet import extract_features
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter
import os

import sys

def resource_path(relative_path):
    """兼容 PyInstaller 打包后的资源路径"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)

save_path = "example_result.xlsx"

def load_anp_emotion_dict(txt_path):
    """加载 ANP → 情绪得分字典的嵌套结构"""
    emotion_dict = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        current_anp = None
        for line in f:
            line = line.strip()
            if line.startswith("ANP:"):
                current_anp = line.split("ANP:")[1].strip()
                emotion_dict[current_anp] = {}
            elif current_anp and ":" in line:
                try:
                    emotion, value = line.split(":")
                    emotion = emotion.strip()
                    value = float(value.strip())
                    emotion_dict[current_anp][emotion] = value
                except ValueError:
                    continue
    return emotion_dict


def get_max_emotion_score_for_anp(anp_name, emotion_dict):
    """返回某个ANP在情绪分数中得分最高的 (情绪, 分数)"""
    anp_key = anp_name.replace(" ", "_")
    if anp_key in emotion_dict:
        emotion_scores = emotion_dict[anp_key]
        if emotion_scores:
            return max(emotion_scores.items(), key=lambda x: x[1])
    return ("N/A", None)



def save_to_excel(image_name, anp_results, iqa_score, nima_mean, nima_std, max_emotions, excel_path=save_path):
    iqa_keys = ["quality", "brightness", "noisiness", "colorfullness", "sharpness"]
    nima_keys = ["NIMA mean", "NIMA std"]

    row_data = [image_name]
    for anp, score, emotion, emo_score in max_emotions:
        row_data.extend([anp, round(float(score), 4), emotion, round(float(emo_score), 4) if emo_score is not None else ""])

    for key in iqa_keys:
        value = iqa_score.get(key, "")
        if hasattr(value, 'item'):  # tensor 转 float
            value = value.item()
        row_data.append(round(float(value), 4) if value != "" else "")

    row_data.append(round(float(nima_mean), 4))
    row_data.append(round(float(nima_std), 4))

    if os.path.exists(excel_path):
        wb = load_workbook(excel_path)
        sheet = wb.active
    else:
        wb = Workbook()
        sheet = wb.active
        # 表头初始化
        header = ["Image"]
        for i in range(len(anp_results[:10])):  # 限制为10个
            header.extend([f"ANP {i+1}", f"Score {i+1}", f"Top Emotion {i+1}", f"Emotion Score {i+1}"])
        header.extend(iqa_keys)
        header.extend(nima_keys)
        sheet.append(header)

    sheet.append(row_data)
    wb.save(excel_path)

def get_all_max_emotions_for_top_anps(anp_results, emotion_dict):
    """处理每个返回的ANP，获取其情绪得分最高项"""
    results = []
    for anp, anp_score in anp_results[:10]:
        emotion, score = get_max_emotion_score_for_anp(anp, emotion_dict)
        results.append((anp, anp_score, emotion, score))
    return results


# img_dir = os.path.join(os.getcwd(), "imgs")
img_dir = resource_path("imgs")
anp_path = resource_path("3244ANPs.txt")


jpg_files = glob.glob(os.path.join(img_dir, "*.jpg"))
jpg_files.sort(key=lambda x: os.path.basename(x))

emotion_scores_dict = load_anp_emotion_dict(resource_path("anp_emotion_scores.txt"))


for jpg in jpg_files:
    print(f"NOW PROCESSING {[os.path.basename(jpg)]}")

    anp_results = match_anp_with_image(jpg, anp_file=anp_path)
    max_emotions = get_all_max_emotions_for_top_anps(anp_results, emotion_scores_dict)

    # for desc, score in anp_results:
    #     print(f"{desc:30s} → original score: {score:.4f}")
    # print()

    print("Top 10 ANPs and their most intense emotions:")
    for anp, anp_score, emotion, emotion_score in max_emotions:
        print(f"{anp:30s}: original score: {anp_score:.4f} → {emotion:10s}: {emotion_score:.4f}")
    print()

    #
    iqa_score = run_clip_iqa(jpg)
    iqa_score = {k: float(v) for k, v in iqa_score.items()}  # 转换 tensor → float
    print("CLIP-IQA score:")
    for k, v in iqa_score.items():
        print(f"{k}: {v}")
    print()

    nima_mean, nima_std = extract_features(jpg, resize=True)
    print("NIMA Score : %0.3f +- (%0.3f)" % (nima_mean, nima_std))
    print("=" * 60)
    print()


    # save_to_excel(os.path.basename(jpg), anp_results, iqa_score, nima_mean, nima_std)
    save_to_excel(os.path.basename(jpg), anp_results, iqa_score, nima_mean, nima_std, max_emotions)

