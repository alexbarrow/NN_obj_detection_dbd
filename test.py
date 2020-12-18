import os
import random
import cv2
import numpy as np

# name = "dbd_img"
# ext = "jpg"
# i = 319
# dir = 'data/temp_data'
# for file in os.listdir(dir):
#     # r = random.randint(0, 500)
#     os.rename(f'{dir}/{file}', f'{dir}/{name}{i}.{ext}')
#     i += 1

# img_path = 'data/test6.jpg'
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(img.astype(np.float32))

# A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

# a = [[1, 4], [3,2], [0, 5]]
# print([item for sublist in a for item in sublist])
# edicts = {'loss_total': 0, 'lr': 1, 'train_time': 2}
# for key, value in dicts.items():
#     valu
acc_list = [10, 20, 30]
for key, value in enumerate(acc_list):
    print(str(key), value)