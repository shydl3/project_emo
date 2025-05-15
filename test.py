# import os
# import glob

# img_dir = os.path.join(os.getcwd(), "imgs")

# dir_list = [f for f in os.listdir(img_dir)
#         if os.path.isdir(os.path.join(img_dir, f))]

# for sub_dir in dir_list:
#         full_dir = os.path.join(img_dir, sub_dir)
#         jpg_files = glob.glob(os.path.join(full_dir, "*.jpg"))
#         jpg_files.sort(key=lambda  x: os.path.basename(x))

#         # print(jpg_files)
import numpy as np

results = [{'image': '1-1.png', 'emotion_score': 0.7881384896125772, 'iqa_avg': 0.7430535316467285, 'nima_mean': np.float64(5.381305513903499), 'nima_std': np.float64(1.4552689722408014)}, {'image': '1-2.png', 'emotion_score': 0.81809687667495, 'iqa_avg': 0.7738005757331848, 'nima_mean': np.float64(5.634783583460376), 'nima_std': np.float64(1.449537056119531)}, {'image': '2-1.png', 'emotion_score': 0.6168211061665607, 'iqa_avg': 0.7536768972873688, 'nima_mean': np.float64(5.345454055815935), 'nima_std': np.float64(1.432511716038948)}, {'image': '2-2.png', 'emotion_score': 0.7163412743163238, 'iqa_avg': 0.8255494415760041, 'nima_mean': np.float64(6.2151692933402956), 'nima_std': np.float64(1.4249528804276905)}, {'image': '3-1.png', 'emotion_score': 0.6790068206679106, 'iqa_avg': 0.700484037399292, 'nima_mean': np.float64(5.740438592154533), 'nima_std': np.float64(1.510094535507233)}, {'image': '3-2.png', 'emotion_score': 0.6821119133752913, 'iqa_avg': 0.7068442106246948, 'nima_mean': np.float64(5.620201406534761), 'nima_std': np.float64(1.5588390439106885)}, {'image': '4-1.png', 'emotion_score': 0.6454873159682898, 'iqa_avg': 0.5219597689807415, 'nima_mean': np.float64(4.391556851333007), 'nima_std': np.float64(1.4005775839870904)}, {'image': '4-2.png', 'emotion_score': 0.704655398343522, 'iqa_avg': 0.6317567884922027, 'nima_mean': np.float64(5.702394957188517), 'nima_std': np.float64(1.4497744323586685)}, {'image': '5-1.png', 'emotion_score': 0.7208945046596434, 'iqa_avg': 0.4667978256940842, 'nima_mean': np.float64(5.036321319639683), 'nima_std': np.float64(1.5006283254597736)}, {'image': '5-2.png', 'emotion_score': 0.8199118099887345, 'iqa_avg': 0.721049427986145, 'nima_mean': np.float64(5.573452933691442), 'nima_std': np.float64(1.5070526068585912)}, {'image': '6-1.png', 'emotion_score': 0.7416752042072284, 'iqa_avg': 0.6366603046655654, 'nima_mean': np.float64(5.09151179343462), 'nima_std': np.float64(1.3952133696858755)}, {'image': '6-2.png', 'emotion_score': 0.6870984101501811, 'iqa_avg': 0.4983398288488388, 'nima_mean': np.float64(5.676463022362441), 'nima_std': np.float64(1.5236736944609497)}, {'image': '7-1.png', 'emotion_score': 0.6737409511385198, 'iqa_avg': 0.7720617413520813, 'nima_mean': np.float64(5.489706008695066), 'nima_std': np.float64(1.3610951243593055)}, {'image': '7-2.png', 'emotion_score': 0.7965277963695365, 'iqa_avg': 0.39674051105976105, 'nima_mean': np.float64(5.781054180813953), 'nima_std': np.float64(1.4249571321511045)}]
# for i in results:
#     print(i)
    # print(i['image'])
    # print(i['emotion_score'])
    # print(i['iqa_avg'])
    # print(i['nima_mean'])
    # print(i['nima_std'])
    # print()

avg_emotion_score = np.mean([item['emotion_score'] for item in results])
avg_iqa = np.mean([item['iqa_avg'] for item in results])
avg_nima_mean = np.mean([item['nima_mean'] for item in results])
avg_nima_std = np.mean([item['nima_std'] for item in results])

print(f"Average Emotion Score: {avg_emotion_score:.4f}")
print(f"Average IQA Score: {avg_iqa:.4f}")
print(f"Average NIMA Mean: {avg_nima_mean:.4f}")
print(f"Average NIMA Std: {avg_nima_std:.4f}")