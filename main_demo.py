import os
import glob
from run_anp_matching import match_anp_with_image
from run_clip_iqa import run_clip_iqa
from nima.evaluate_inception_resnet import extract_features


img_dir = os.path.join(os.getcwd(), "imgs")
# print(img_dir)

jpg_files = glob.glob(os.path.join(img_dir, "*.jpg"))
jpg_files.sort(key=lambda x: os.path.basename(x))

for jpg in jpg_files:
    print(f"NOW PROCESSING {[os.path.basename(jpg)]}")

    anp_result = match_anp_with_image(jpg)
    for desc, score in anp_result:
        print(f"{desc:30s} → original score: {score:.4f}")
    print()

    iqa_score = run_clip_iqa(jpg)
    print("CLIP-IQA 打分结果:")
    for k, v in iqa_score.items():
        print(f"{k:15s}: {v:.4f}")
    print()

    nima_mean, nima_std = extract_features(jpg, resize=True)
    print("NIMA Score : %0.3f +- (%0.3f)" % (nima_mean, nima_std))
    print("=" * 60)
    print()






