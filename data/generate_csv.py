import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




labels_e = ['Broken soybeans', 'Immature soybeans', 'Intact soybeans', 'Skin-damaged soybeans', 'Spotted soybeans']
labels_cn = ['破碎的大豆', '未成熟大豆', '完整大豆', '皮肤受损的大豆', '大豆泥斑']
labels = ['0', '1', '2', '3', '4']

root_path = './'
all_names = [f for f in os.listdir(root_path) if os.path.isdir(f)]


image_id=[]
all_label=[]
all_label_cn =[]
for class_folder in all_names:
    label = labels[labels_e.index(class_folder)]
    label_cn = labels_cn[labels_e.index(class_folder)]
    class_path = os.path.join(root_path, class_folder)

    images = os.listdir(class_path)
    for i, path in enumerate(images):
        images[i] = str(class_folder) + '/' + path

    images_len = len(images)
    images_label = np.repeat(label, images_len)
    images_label_cn = np.repeat(label_cn, images_len)

    image_id.extend(images)
    all_label.extend(images_label)
    all_label_cn.extend(images_label_cn)


name = root_path + 'train.csv'
df = pd.DataFrame({'image_id': image_id, 'label': all_label})
df.to_csv(name, index=False)

